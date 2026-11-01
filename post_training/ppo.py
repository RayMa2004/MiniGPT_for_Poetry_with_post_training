"""Experimental remote-only on-policy PPO with a fixed, validated AI reward model.

There is no import-time model construction or checkpoint loading. Actor exports retain
exactly the baseline MiniGPT keys; the value head lives only in trainer_state.pt.
"""
import argparse
import copy
import math
import random
from .core import (resolve, load_config, read_json, read_jsonl, file_hash,
                   validate_task, validate_group_splits)

GROUPS = tuple((mode, form) for mode in ('prefix', 'free') for form in ('wujue', 'qijue'))


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_reward_gate(report, checkpoint, vocab, config, payload=None):
    """Check the report before model loading, then bind its metrics to loaded weights."""
    if (report.get('kind') != 'reward_validation_v1' or report.get('passed') is not True
            or report.get('reward_checkpoint_sha256') != file_hash(checkpoint)
            or report.get('vocab_sha256') != file_hash(vocab)):
        raise ValueError('PPO requires a passing report bound to the exact reward checkpoint and vocabulary')
    settings = config['reward']
    minimum_count, minimum_accuracy = settings['minimum_val_pairs'], settings['minimum_val_accuracy']
    if (type(minimum_count) is not int or minimum_count < 1 or not _finite_number(minimum_accuracy)
            or not .5 < minimum_accuracy <= 1):
        raise ValueError('Invalid current reward validation thresholds')
    count, accuracy = report.get('val_pair_count'), report.get('val_accuracy')
    if (type(count) is not int or count < minimum_count or not _finite_number(accuracy)
            or not minimum_accuracy <= accuracy <= 1):
        raise ValueError('Reward validation fails current aggregate thresholds')
    groups = report.get('by_group', {})
    if not isinstance(groups, dict) or set(groups) != {m + '/' + f for m, f in GROUPS}:
        raise ValueError('Reward validation must cover all four mode/form groups')
    total, correct = 0, 0.
    for group in groups.values():
        n, acc = group.get('count'), group.get('accuracy')
        if type(n) is not int or n <= 0 or not _finite_number(acc) or not .5 < acc <= 1:
            raise ValueError('Every reward validation group must beat chance with nonempty evidence')
        total += n
        correct += n * acc
    if total != count or not math.isclose(correct / total, accuracy, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError('Reward validation group counts/accuracy disagree with aggregate')
    normalization = report.get('normalization', {})
    if (not isinstance(normalization, dict) or set(normalization) != {'mean', 'std'}
            or not _finite_number(normalization['mean']) or not _finite_number(normalization['std'])
            or normalization['std'] <= 1e-6):
        raise ValueError('Reward normalization must be finite and use nondegenerate held-out validation scores')
    if payload is not None:
        if (payload.get('kind') != 'post_training_reward_v1'
                or payload.get('vocab_sha256') != file_hash(vocab)
                or payload.get('reward_normalization') != normalization):
            raise ValueError('Reward weights and report normalization/identity disagree')
        validation = payload.get('validation')
        if (not isinstance(validation, dict)
                or not {'val_pair_count', 'val_accuracy', 'by_group', 'normalization'} <= validation.keys()
                or any(report.get(key) != value for key, value in validation.items())):
            raise ValueError('Reward checkpoint validation metrics disagree with its identity-bound report')
    return dict(normalization)


def validate_ppo_settings(config):
    from .learning import validate_hyperparameters
    validate_hyperparameters(config, 'ppo')
    t = config['ppo']
    for name in ('gamma', 'gae_lambda'):
        if not _finite_number(t[name]) or not 0 <= t[name] <= 1:
            raise ValueError(f'ppo.{name} must be in [0,1]')
    for name in ('clip_range', 'value_clip_range', 'target_kl', 'reward_clip'):
        if not _finite_number(t[name]) or t[name] <= 0:
            raise ValueError(f'ppo.{name} must be finite positive')
    if t['clip_range'] >= 1:
        raise ValueError('ppo.clip_range must be less than one')
    for name in ('value_coefficient', 'entropy_coefficient', 'kl_coefficient'):
        if not _finite_number(t[name]) or t[name] < 0:
            raise ValueError(f'ppo.{name} must be finite nonnegative')
    limit = t.get('reference_kl_limit', .2)
    if not _finite_number(limit) or limit <= 0:
        raise ValueError('ppo.reference_kl_limit must be finite positive')
    snapshot_every = t.get('snapshot_every', 10)
    if type(snapshot_every) is not int or snapshot_every <= 0:
        raise ValueError('ppo.snapshot_every must be a positive integer')
    if t['rollout_batch_size'] < len(GROUPS) or t['rollout_batch_size'] % len(GROUPS):
        raise ValueError('PPO rollout_batch_size must be a positive multiple of four for balanced coverage')


class BalancedTasks:
    """Shuffle and cycle independently within each group; rotate balanced groups."""
    def __init__(self, rows, seed):
        self.random = random.Random(seed)
        self.groups = {group: [] for group in GROUPS}
        for row in rows:
            validate_task(row)
            if row['split'] != 'train':
                raise ValueError('PPO task scheduler accepts only training split')
            self.groups[row['mode'], row['form']].append(row)
        if any(not rows for rows in self.groups.values()):
            raise ValueError('PPO needs training tasks for all four mode/form groups')
        self.offsets = {group: 0 for group in GROUPS}
        self.order = []
        for rows in self.groups.values():
            self.random.shuffle(rows)

    def take(self, size):
        result = []
        for _ in range(size):
            if not self.order:
                self.order = list(GROUPS)
                self.random.shuffle(self.order)
            group = self.order.pop()
            rows, index = self.groups[group], self.offsets[group]
            if index == len(rows):
                self.random.shuffle(rows)
                index = 0
            result.append(rows[index])
            self.offsets[group] = index + 1
        return result


def legal_token_ids(tokenizer, device):
    import torch
    from evaluation.metrics import is_han
    allowed = sorted(index for text, index in tokenizer.stoi.items() if len(text) == 1 and is_han(text))
    if len(allowed) < 2:
        raise ValueError('PPO requires at least two legal Han tokens')
    return torch.tensor(allowed, dtype=torch.long, device=device)


def action_statistics(logits, actions, legal_ids):
    """Temperature exactly one, identical fixed Han support for every policy."""
    import torch
    legal_logits = logits.index_select(-1, legal_ids).float()
    if not torch.isfinite(legal_logits).all():
        raise RuntimeError('Nonfinite legal action logits')
    indices = torch.searchsorted(legal_ids, actions)
    safe = indices.clamp(max=legal_ids.numel() - 1)
    if not torch.equal(legal_ids[safe], actions):
        raise ValueError('Action outside the shared legal Han support')
    log_distribution = torch.log_softmax(legal_logits, dim=-1)
    if not torch.isfinite(log_distribution).all():
        raise RuntimeError('Nonfinite legal action log distribution')
    logp = log_distribution.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_distribution.exp() * log_distribution).sum(-1)
    return logp, entropy, log_distribution


def evaluate_actions(policy, value_head, trajectory, legal_ids, device):
    import torch
    from .learning import hidden_states
    # eval disables dropout only: actor and value gradients stay enabled here.
    policy.eval()
    value_head.eval()
    ids = torch.tensor([trajectory['ids'][:-1]], dtype=torch.long, device=device)
    positions = torch.tensor(trajectory['positions'], dtype=torch.long, device=device)
    hidden = hidden_states(policy, ids)[0].index_select(0, positions)
    logp, entropy, distribution = action_statistics(policy.lm_head(hidden), trajectory['actions'], legal_ids)
    values = value_head(hidden).squeeze(-1)
    if not torch.isfinite(values).all():
        raise RuntimeError('Nonfinite value estimates')
    return logp, values, entropy, distribution


def rollout_trajectory(policy, reference, value_head, tokenizer, task, legal_ids, generator, device):
    import torch
    from .learning import hidden_states
    validate_task(task)
    if task['split'] != 'train':
        raise ValueError('No validation/test rollout tasks in PPO training')
    if any(ch not in tokenizer.stoi for ch in task['prompt']) or any(ch not in tokenizer.stoi for ch in '，。'):
        raise ValueError('PPO prefix/punctuation must exist in the unchanged vocabulary')
    width = 5 if task['form'] == 'wujue' else 7
    if 4 * (width + 1) + 2 > min(policy.block_size, reference.block_size):
        raise ValueError('Complete PPO poem including EOS exceeds the context window')
    policy.eval()
    reference.eval()
    value_head.eval()
    ids = [tokenizer.bos_id] + [tokenizer.stoi[ch] for ch in task['prompt']]
    positions, actions, old_logprobs, old_values, reference_logprobs = [], [], [], [], []
    with torch.no_grad():
        for line in range(4):
            for column in range(width):
                if line == 0 and column < len(task['prompt']):
                    continue
                context = torch.tensor([ids], dtype=torch.long, device=device)
                features = hidden_states(policy, context)[0, -1]
                logits = policy.lm_head(features)
                legal_logits = logits.index_select(-1, legal_ids).float()
                if not torch.isfinite(legal_logits).all():
                    raise RuntimeError('Nonfinite rollout logits')
                probabilities = torch.softmax(legal_logits, dim=-1)
                legal_index = torch.multinomial(probabilities, 1, generator=generator)
                action = legal_ids[legal_index].squeeze(0)
                old_logp, _, _ = action_statistics(logits, action, legal_ids)
                reference_features = hidden_states(reference, context)[0, -1]
                reference_logp, _, _ = action_statistics(reference.lm_head(reference_features), action, legal_ids)
                value = value_head(features).squeeze(-1)
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite rollout value')
                positions.append(len(ids) - 1)
                actions.append(action)
                old_logprobs.append(old_logp)
                old_values.append(value)
                reference_logprobs.append(reference_logp)
                ids.append(int(action.item()))
            ids.append(tokenizer.stoi['，' if line % 2 == 0 else '。'])
        ids.append(tokenizer.eos_id)
    return dict(task=task, ids=ids, text=tokenizer.decode(ids), positions=positions,
                actions=torch.stack(actions).detach(), old_logprobs=torch.stack(old_logprobs).detach(),
                old_values=torch.stack(old_values).detach(), reference_logprobs=torch.stack(reference_logprobs).detach())


def terminal_gae(rewards, values, gamma, gae_lambda):
    """One poem is one terminal episode; forced tokens introduce no actions."""
    import torch
    if rewards.ndim != 1 or values.shape != rewards.shape or rewards.numel() == 0:
        raise ValueError('GAE needs matching nonempty action vectors')
    advantages = torch.zeros_like(values)
    following_value, gae = values.new_zeros(()), values.new_zeros(())
    for index in range(len(values) - 1, -1, -1):
        delta = rewards[index] + gamma * following_value - values[index]
        gae = delta + gamma * gae_lambda * gae
        advantages[index] = gae
        following_value = values[index]
    return advantages.detach(), (advantages + values).detach()


def clipped_losses(new_logp, old_logp, advantages, values, old_values, returns, clip_range, value_clip_range):
    import torch
    ratio = (new_logp - old_logp).exp()
    actor = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages).mean()
    clipped_values = old_values + (values - old_values).clamp(-value_clip_range, value_clip_range)
    critic = .5 * torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    return actor, critic


def assign_rewards(trajectories, reward, tokenizer, normalization, settings, device):
    import torch
    from .learning import full_batch
    rows = [dict(text=t['text'], prompt=t['task']['prompt']) for t in trajectories]
    ids, lengths = full_batch(rows, tokenizer, reward.backbone.block_size, device)
    reward.eval()
    with torch.no_grad():
        raw_rewards = reward(ids, lengths)
        normalized = ((raw_rewards - normalization['mean']) / normalization['std']).clamp(
            -settings['reward_clip'], settings['reward_clip'])
        if not torch.isfinite(raw_rewards).all() or not torch.isfinite(normalized).all():
            raise RuntimeError('Nonfinite reward scores')
        for trajectory, scalar, raw in zip(trajectories, normalized, raw_rewards):
            shaping = -settings['kl_coefficient'] * (trajectory['old_logprobs'] - trajectory['reference_logprobs'])
            rewards = shaping.clone()
            rewards[-1] += scalar
            advantage, returns = terminal_gae(rewards, trajectory['old_values'], settings['gamma'], settings['gae_lambda'])
            trajectory.update(advantages=advantage, returns=returns, reward=float(scalar.item()), raw_reward=float(raw.item()))
        flat = torch.cat([t['advantages'] for t in trajectories])
        mean, std = flat.mean(), flat.std(unbiased=False)
        for trajectory in trajectories:
            trajectory['advantages'] = ((trajectory['advantages'] - mean) / (std + 1e-8)).detach()


def policy_drift(policy, reference, value_head, trajectories, legal_ids, device):
    import torch
    total, old_kl, reference_kl = 0, 0., 0.
    with torch.no_grad():
        for trajectory in trajectories:
            logp, _, _, distribution = evaluate_actions(policy, value_head, trajectory, legal_ids, device)
            _, _, _, ref_distribution = evaluate_actions(reference, value_head, trajectory, legal_ids, device)
            difference = logp - trajectory['old_logprobs']
            # Nonnegative old->current estimator; exact current->reference categorical KL
            # at rollout states uses the entire shared Han support.
            old_kl += float((difference.exp() - 1 - difference).sum().item())
            reference_kl += float((distribution.exp() * (distribution - ref_distribution)).sum().item())
            total += logp.numel()
    result = dict(old_policy_kl=old_kl / total, reference_kl=reference_kl / total)
    if any(not math.isfinite(value) for value in result.values()):
        raise RuntimeError('Nonfinite KL drift')
    result['reference_kl'] = max(0., result['reference_kl'])
    return result


def optimize_rollout(policy, reference, value_head, optimizer, trajectories, legal_ids, settings, device):
    import torch
    total_actions = sum(len(t['positions']) for t in trajectories)
    parameters = list(policy.parameters()) + list(value_head.parameters())
    last = dict(actor_loss=0., value_loss=0., entropy=0.)
    completed = 0
    drift = policy_drift(policy, reference, value_head, trajectories, legal_ids, device)
    for _ in range(settings['ppo_epochs']):
        if (drift['old_policy_kl'] > settings['target_kl']
                or drift['reference_kl'] > settings.get('reference_kl_limit', .2)):
            break
        optimizer.zero_grad(set_to_none=True)
        last = dict(actor_loss=0., value_loss=0., entropy=0.)
        # Accumulate globally action-weighted gradients without retaining every poem graph.
        for trajectory in trajectories:
            logp, values, entropy, _ = evaluate_actions(policy, value_head, trajectory, legal_ids, device)
            actor, critic = clipped_losses(logp, trajectory['old_logprobs'], trajectory['advantages'], values,
                                           trajectory['old_values'], trajectory['returns'],
                                           settings['clip_range'], settings['value_clip_range'])
            fraction = logp.numel() / total_actions
            entropy_mean = entropy.mean()
            loss = fraction * (actor + settings['value_coefficient'] * critic - settings['entropy_coefficient'] * entropy_mean)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite PPO objective')
            loss.backward()
            for name, value in [('actor_loss', actor), ('value_loss', critic), ('entropy', entropy_mean)]:
                last[name] += fraction * float(value.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, settings['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise RuntimeError('Nonfinite PPO parameters after update')
        completed += 1
        last['gradient_norm'] = float(gradient_norm.item())
        drift = policy_drift(policy, reference, value_head, trajectories, legal_ids, device)
    return dict(last, **drift, ppo_epochs_completed=completed, action_count=total_actions,
                reference_stop=drift['reference_kl'] > settings.get('reference_kl_limit', .2),
                epoch_kl_stop=drift['old_policy_kl'] > settings['target_kl'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', required=True, help='Task JSONL; only split=train is used')
    parser.add_argument('--init-checkpoint', required=True, help='Explicit exported RSFT policy, never the baseline default')
    parser.add_argument('--reward-checkpoint', required=True)
    parser.add_argument('--reward-validation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    task_path, initial_path = resolve(args.tasks), resolve(args.init_checkpoint)
    reward_path, report_path = resolve(args.reward_checkpoint), resolve(args.reward_validation)
    vocab = resolve(config['paths']['vocab'])
    report = read_json(report_path)
    normalization = validate_reward_gate(report, reward_path, vocab, config)
    rows = read_jsonl(task_path)
    for row in rows:
        validate_task(row)
    validate_group_splits(rows)
    tasks = [row for row in rows if row['split'] == 'train']
    scheduler = BalancedTasks(tasks, config['seed'])
    # All neural imports, weight deserialization and device work require explicit main().
    import torch
    from checkpoint import read_checkpoint, save_checkpoint, capture_rng
    from .learning import seed_all, load_policy, load_reward, export_policy, TrainingRun
    validate_ppo_settings(config)
    initial_payload = read_checkpoint(initial_path)
    if (initial_payload.get('post_training', {}).get('stage') not in ('rsft', 'sft')
            or initial_payload.get('data_identity', {}).get('vocab', {}).get('sha256') != file_hash(vocab)):
        raise ValueError('--init-checkpoint must be a vocabulary-bound RSFT policy export')
    del initial_payload
    seed_all(config['seed'])
    settings = config['ppo']
    policy, tokenizer, model_config = load_policy(initial_path, vocab, args.device)
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reward, reward_payload = load_reward(reward_path, tokenizer, vocab, args.device)
    validate_reward_gate(report, reward_path, vocab, config, reward_payload)
    del reward_payload
    if min(policy.block_size, reward.backbone.block_size) < 34:
        raise ValueError('PPO requires room for the complete seven-character quatrain including BOS/EOS')
    legal_ids = legal_token_ids(tokenizer, args.device)
    # Fail on every configured prompt before the first rollout, not halfway through training.
    if any(ch not in tokenizer.stoi for task in tasks for ch in task['prompt']) or any(ch not in tokenizer.stoi for ch in '，。'):
        raise ValueError('PPO task prefix or forced punctuation is out of vocabulary')
    value_head = torch.nn.Linear(model_config['n_embd'], 1).to(args.device)
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)
    value_head.eval()
    policy.eval()
    optimizer = torch.optim.AdamW([
        dict(params=list(policy.parameters()), lr=settings['learning_rate']),
        dict(params=list(value_head.parameters()), lr=settings['value_learning_rate'])], weight_decay=0.)
    generator = torch.Generator(device=args.device).manual_seed(config['seed'])
    inputs = dict(tasks=task_path, initial_policy=initial_path, reference_policy=initial_path,
                  reward_checkpoint=reward_path, reward_validation=report_path, vocab=vocab)
    with TrainingRun(args.output, 'ppo', config, inputs) as run:
        run.metadata.update(reference_checkpoint_sha256=file_hash(initial_path),
                            reward_checkpoint_sha256=file_hash(reward_path), reward_validation_sha256=file_hash(report_path),
                            task_sha256=file_hash(task_path), vocab_sha256=file_hash(vocab), normalization=normalization,
                            behavior_sampling=dict(temperature=1., top_k=None, top_p=None, e05=False,
                                                   support='fixed Han vocabulary', forced_punctuation='，。，。', forced_eos=True),
                            selection='last; no independent PPO validation or best-policy claim',
                            reward_evidence='AI preference validation; training reward is not independent poetry quality evidence')
        for update in range(1, settings['updates'] + 1):
            trajectories = [rollout_trajectory(policy, reference, value_head, tokenizer, task, legal_ids, generator, args.device)
                            for task in scheduler.take(settings['rollout_batch_size'])]
            assign_rewards(trajectories, reward, tokenizer, normalization, settings, args.device)
            metrics = optimize_rollout(policy, reference, value_head, optimizer, trajectories, legal_ids, settings, args.device)
            metrics.update(update=update, normalized_reward=sum(t['reward'] for t in trajectories) / len(trajectories),
                           raw_reward=sum(t['raw_reward'] for t in trajectories) / len(trajectories))
            run.log(metrics)
            metadata = dict(stage='ppo', update=update, reference_checkpoint_sha256=file_hash(initial_path),
                            reward_checkpoint_sha256=file_hash(reward_path), reward_validation_sha256=file_hash(report_path),
                            task_sha256=file_hash(task_path), normalization=normalization, selection='last',
                            experimental=True, independent_quality_validation=False)
            export_policy(run.path / 'last.pt', policy, model_config, vocab, metadata)
            snapshot_every = settings.get('snapshot_every', 10)
            if update % snapshot_every == 0:
                export_policy(run.path / f'policy_update_{update:06d}.pt', policy, model_config, vocab, metadata)
            save_checkpoint(run.path / 'trainer_state.pt', dict(kind='post_training_ppo_state_v1', update=update,
                            optimizer_state_dict=optimizer.state_dict(), value_head_state_dict=value_head.state_dict(),
                            rng=capture_rng(), rollout_generator_state=generator.get_state(),
                            restart_policy='Diagnostics only; no exact resume supported; warm-start a new explicit run.',
                            actor_checkpoint_sha256=file_hash(run.path / 'last.pt'), metadata=metadata))
            print(f"PPO update={update} actions={metrics['action_count']} old_kl={metrics['old_policy_kl']:.5f} reference_kl={metrics['reference_kl']:.5f}", flush=True)
            if metrics['reference_stop']:
                run.metadata['stop_reason'] = 'reference_kl_limit'
                break
        run.metadata.update(updates_completed=update, final_policy='last.pt',
                            next_step='Independent held-out blind poetry evaluation under the same E05+E06 decoding conditions.')


if __name__ == '__main__':
    main()


