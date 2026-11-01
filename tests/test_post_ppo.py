"""Remote-only PPO contracts. Intentionally not executed during local authoring."""
import copy
import json
import tempfile
import unittest
from pathlib import Path


class PPOContracts(unittest.TestCase):
    def test_terminal_gae_has_no_bootstrap_across_poems(self):
        import torch
        from post_training.ppo import terminal_gae
        advantages, returns = terminal_gae(torch.tensor([0., 2.]), torch.tensor([.5, .25]), 1., 1.)
        torch.testing.assert_close(advantages, torch.tensor([1.5, 1.75]))
        torch.testing.assert_close(returns, torch.tensor([2., 2.]))
        one, _ = terminal_gae(torch.tensor([3.]), torch.tensor([.5]), .9, .8)
        torch.testing.assert_close(one, torch.tensor([2.5]))

    def test_clipped_actor_and_value_objectives(self):
        import torch
        from post_training.ppo import clipped_losses
        actor, critic = clipped_losses(torch.tensor([2., .5]).log(), torch.zeros(2),
            torch.tensor([1., -1.]), torch.tensor([2., 2.]), torch.zeros(2), torch.ones(2), .2, .2)
        self.assertAlmostEqual(actor.item(), -.2, places=6)
        self.assertAlmostEqual(critic.item(), .5, places=6)

    def test_mask_support_and_neutral_probabilities(self):
        import torch
        from post_training.ppo import legal_token_ids, action_statistics
        from tokenizer import CharTokenizer
        tokenizer = CharTokenizer.build(['春江，。'])
        legal = legal_token_ids(tokenizer, 'cpu')
        logits = torch.full((1, tokenizer.vocab_size), 100.)
        logits[0, tokenizer.stoi['春']] = 0.
        logits[0, tokenizer.stoi['江']] = 1.
        actions = torch.tensor([tokenizer.stoi['春']])
        logp, entropy, distribution = action_statistics(logits, actions, legal)
        self.assertEqual(set(legal.tolist()), {tokenizer.stoi['春'], tokenizer.stoi['江']})
        self.assertAlmostEqual(logp.item(), -torch.logsumexp(torch.tensor([0., 1.]), 0).item(), places=6)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertEqual(distribution.shape, (1, 2))
        with self.assertRaises(ValueError):
            action_statistics(logits, torch.tensor([tokenizer.eos_id]), legal)

    def make_models(self):
        import torch
        from model import MiniGPT
        from tokenizer import CharTokenizer
        tokenizer = CharTokenizer.build(['春江花月夜山水风云，。'])
        cfg = dict(vocab_size=tokenizer.vocab_size, block_size=40, n_embd=8, n_head=2, n_layer=1, dropout=.7)
        policy = MiniGPT(**cfg)
        critic = torch.nn.Linear(8, 1)
        torch.nn.init.zeros_(critic.weight)
        torch.nn.init.zeros_(critic.bias)
        return policy, copy.deepcopy(policy), critic, tokenizer, cfg

    def test_rollout_alignment_old_logprob_and_dropout(self):
        import torch
        from post_training.ppo import legal_token_ids, rollout_trajectory, evaluate_actions
        policy, reference, critic, tokenizer, _ = self.make_models()
        task = dict(task_id='t', group_id='g', prompt='春江', form='wujue', mode='prefix', split='train')
        legal = legal_token_ids(tokenizer, 'cpu')
        trajectory = rollout_trajectory(policy, reference, critic, tokenizer, task, legal,
                                       torch.Generator().manual_seed(7), 'cpu')
        self.assertFalse(policy.training)
        self.assertTrue(all(not module.training for module in policy.modules()))
        self.assertEqual(trajectory['text'][:2], '春江')
        self.assertEqual([trajectory['text'][i] for i in (5, 11, 17, 23)], ['，', '。', '，', '。'])
        self.assertEqual(trajectory['ids'][-1], tokenizer.eos_id)
        self.assertEqual(len(trajectory['positions']), 18)
        for position, action in zip(trajectory['positions'], trajectory['actions'].tolist()):
            self.assertEqual(trajectory['ids'][position + 1], action)
            self.assertNotIn(tokenizer.itos[action], ['，', '。', '<EOS>'])
        self.assertNotIn(0, trajectory['positions'])
        self.assertNotIn(1, trajectory['positions'])
        logp, values, _, _ = evaluate_actions(policy, critic, trajectory, legal, 'cpu')
        torch.testing.assert_close(logp, trajectory['old_logprobs'], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(values, trajectory['old_values'])
        torch.testing.assert_close(logp, trajectory['reference_logprobs'], rtol=1e-5, atol=1e-6)
        self.assertFalse(trajectory['old_logprobs'].requires_grad)
        loss = -logp.mean() + values.square().mean()
        loss.backward()
        self.assertIsNotNone(policy.token_embedding.weight.grad)

    def test_all_form_mode_rollouts_and_oov_rejection(self):
        import torch
        from post_training.ppo import rollout_trajectory, legal_token_ids
        policy, reference, critic, tokenizer, _ = self.make_models()
        legal = legal_token_ids(tokenizer, 'cpu')
        for form, width in [('wujue', 5), ('qijue', 7)]:
            for prompt in ['', '春江花月']:
                task = dict(task_id='t', group_id='g', prompt=prompt, form=form,
                            mode='prefix' if prompt else 'free', split='train')
                trajectory = rollout_trajectory(policy, reference, critic, tokenizer, task, legal,
                                               torch.Generator().manual_seed(3), 'cpu')
                self.assertEqual(len(trajectory['text']), 4 * (width + 1))
                self.assertEqual(len(trajectory['actions']), 4 * width - len(prompt))
        task.update(prompt='未知', mode='prefix')
        with self.assertRaises(ValueError):
            rollout_trajectory(policy, reference, critic, tokenizer, task, legal,
                               torch.Generator().manual_seed(3), 'cpu')

    def test_balanced_scheduler_covers_four_groups_and_cycles(self):
        from post_training.ppo import BalancedTasks
        rows = [dict(task_id=f'{mode}-{form}', group_id=f'{mode}-{form}', prompt='春' if mode == 'prefix' else '',
                     mode=mode, form=form, split='train') for mode in ['prefix', 'free'] for form in ['wujue', 'qijue']]
        scheduler = BalancedTasks(rows, 9)
        batch = scheduler.take(8)
        counts = {(mode, form): 0 for mode in ['prefix', 'free'] for form in ['wujue', 'qijue']}
        for task in batch:
            counts[task['mode'], task['form']] += 1
        self.assertEqual(set(counts.values()), {2})
        with self.assertRaises(ValueError):
            BalancedTasks(rows[:-1], 9)
        rows[0]['split'] = 'val'
        with self.assertRaises(ValueError):
            BalancedTasks(rows, 9)

    def test_reward_gate_binds_hash_vocabulary_normalization_and_current_threshold(self):
        from post_training.ppo import validate_reward_gate
        from post_training.core import file_hash
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'reward.pt'
            vocab = Path(temporary) / 'vocab.json'
            checkpoint.write_bytes(b'identity-only-test')
            vocab.write_text('{}', encoding='utf-8')
            metrics = dict(val_pair_count=40, val_accuracy=.75,
                           by_group={f'{mode}/{form}':dict(count=10, accuracy=.75) for mode in ['prefix', 'free'] for form in ['wujue', 'qijue']},
                           normalization=dict(mean=1., std=2.))
            report = dict(metrics, kind='reward_validation_v1', passed=True,
                          reward_checkpoint_sha256=file_hash(checkpoint), vocab_sha256=file_hash(vocab))
            config = dict(reward=dict(minimum_val_pairs=30, minimum_val_accuracy=.65))
            payload = dict(kind='post_training_reward_v1', vocab_sha256=file_hash(vocab),
                           validation=metrics, reward_normalization=metrics['normalization'])
            validate_reward_gate(report, checkpoint, vocab, config, payload)
            for change in [dict(passed=False), dict(reward_checkpoint_sha256='0'*64),
                           dict(vocab_sha256='0'*64), dict(val_pair_count=2), dict(val_accuracy=.5),
                           dict(normalization=dict(mean=1., std=0.))]:
                with self.assertRaises(ValueError):
                    validate_reward_gate(dict(report, **change), checkpoint, vocab, config, payload)
            bad = copy.deepcopy(payload)
            bad['validation']['by_group']['free/qijue']['accuracy'] = .5
            with self.assertRaises(ValueError):
                validate_reward_gate(report, checkpoint, vocab, config, bad)
            with self.assertRaises(ValueError):
                validate_reward_gate(report, checkpoint, vocab,
                                     dict(reward=dict(minimum_val_pairs=50, minimum_val_accuracy=.65)), payload)

    def test_actor_export_strict_compatibility_has_no_critic_keys(self):
        import torch
        from post_training.learning import export_policy
        from checkpoint import read_checkpoint, load_model_checkpoint
        policy, _, critic, tokenizer, cfg = self.make_models()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocab = root / 'vocab.json'
            tokenizer.save(vocab)
            target = root / 'last.pt'
            export_policy(target, policy, cfg, vocab, dict(stage='ppo'))
            payload = read_checkpoint(target)
            self.assertEqual(set(payload['model_state_dict']), set(policy.state_dict()))
            self.assertFalse(any('value_head' in key or 'critic' in key for key in payload['model_state_dict']))
            loaded, _ = load_model_checkpoint(target, tokenizer, torch.device('cpu'), vocab_path=vocab)
            loaded.load_state_dict(policy.state_dict(), strict=True)

    def test_extreme_finite_logits_cannot_create_nonfinite_log_distribution(self):
        import torch
        from post_training.ppo import action_statistics
        logits = torch.tensor([[3e38, -3e38]])
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(RuntimeError):
            action_statistics(logits, torch.tensor([0]), torch.tensor([0, 1]))

    def test_rollout_batch_size_must_be_a_multiple_of_four(self):
        from post_training.core import load_config
        from post_training.ppo import validate_ppo_settings
        config = load_config()
        for size in [1, 5, 6, 7]:
            config['ppo']['rollout_batch_size'] = size
            with self.assertRaises(ValueError):
                validate_ppo_settings(config)
        config['ppo']['rollout_batch_size'] = 8
        validate_ppo_settings(config)

    def test_raw_nonfinite_reference_kl_is_not_hidden_by_clamp(self):
        import torch
        from unittest.mock import patch
        from post_training.ppo import policy_drift
        trajectory = dict(old_logprobs=torch.zeros(1))
        finite = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros((1, 1)))
        bad_reference = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.full((1, 1), float('nan')))
        with patch('post_training.ppo.evaluate_actions', side_effect=[finite, bad_reference]):
            with self.assertRaises(RuntimeError):
                policy_drift(None, None, None, [trajectory], None, 'cpu')

    def test_actual_optimizer_updates_actor_and_separate_critic(self):
        import torch
        from post_training.ppo import legal_token_ids, rollout_trajectory, optimize_rollout
        policy, reference, critic, tokenizer, _ = self.make_models()
        legal = legal_token_ids(tokenizer, 'cpu')
        task = dict(task_id='t', group_id='g', prompt='春江', form='wujue', mode='prefix', split='train')
        trajectory = rollout_trajectory(policy, reference, critic, tokenizer, task, legal,
                                       torch.Generator().manual_seed(17), 'cpu')
        trajectory.update(advantages=torch.linspace(-1., 1., len(trajectory['positions'])),
                          returns=torch.ones_like(trajectory['old_values']))
        actor_before = policy.token_embedding.weight.detach().clone()
        old_logp_before = trajectory['old_logprobs'].clone()
        reference_before = {key: value.detach().clone() for key, value in reference.state_dict().items()}
        optimizer = torch.optim.AdamW([dict(params=policy.parameters(), lr=.001),
                                       dict(params=critic.parameters(), lr=.01)], weight_decay=0.)
        settings = dict(ppo_epochs=1, clip_range=.2, value_clip_range=.2, value_coefficient=.5,
                        entropy_coefficient=0., grad_clip=1., target_kl=100., reference_kl_limit=100.)
        metrics = optimize_rollout(policy, reference, critic, optimizer, [trajectory], legal, settings, 'cpu')
        self.assertEqual(metrics['ppo_epochs_completed'], 1)
        self.assertFalse(torch.equal(actor_before, policy.token_embedding.weight))
        self.assertGreater(abs(critic.bias.item()), 0.)
        self.assertFalse(policy.training)
        torch.testing.assert_close(trajectory['old_logprobs'], old_logp_before, rtol=0, atol=0)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, reference_before[key], rtol=0, atol=0)

    def test_reference_drift_stops_before_optimizer_changes_parameters(self):
        import torch
        from post_training.ppo import legal_token_ids, rollout_trajectory, optimize_rollout
        policy, reference, critic, tokenizer, _ = self.make_models()
        legal = legal_token_ids(tokenizer, 'cpu')
        with torch.no_grad():
            reference.lm_head.weight[legal[0], 0].add_(1000.)
        task = dict(task_id='t', group_id='g', prompt='', form='wujue', mode='free', split='train')
        trajectory = rollout_trajectory(policy, reference, critic, tokenizer, task, legal,
                                       torch.Generator().manual_seed(21), 'cpu')
        trajectory.update(advantages=torch.ones_like(trajectory['old_values']),
                          returns=torch.ones_like(trajectory['old_values']))
        actor_before = policy.token_embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW([dict(params=policy.parameters(), lr=.001),
                                       dict(params=critic.parameters(), lr=.01)], weight_decay=0.)
        settings = dict(ppo_epochs=2, clip_range=.2, value_clip_range=.2, value_coefficient=.5,
                        entropy_coefficient=0., grad_clip=1., target_kl=.1, reference_kl_limit=1e-10)
        metrics = optimize_rollout(policy, reference, critic, optimizer, [trajectory], legal, settings, 'cpu')
        self.assertTrue(metrics['reference_stop'])
        self.assertEqual(metrics['ppo_epochs_completed'], 0)
        self.assertEqual(len(optimizer.state), 0)
        torch.testing.assert_close(actor_before, policy.token_embedding.weight, rtol=0, atol=0)

    def test_old_policy_kl_stops_before_any_optimizer_step(self):
        import torch
        from unittest.mock import patch
        from post_training.ppo import legal_token_ids, rollout_trajectory, optimize_rollout
        policy, reference, critic, tokenizer, _ = self.make_models()
        legal = legal_token_ids(tokenizer, 'cpu')
        task = dict(task_id='t', group_id='g', prompt='春江', form='wujue', mode='prefix', split='train')
        trajectory = rollout_trajectory(policy, reference, critic, tokenizer, task, legal,
                                       torch.Generator().manual_seed(31), 'cpu')
        trajectory.update(advantages=torch.ones_like(trajectory['old_values']),
                          returns=torch.ones_like(trajectory['old_values']))
        # policy_drift uses d=new_logp-old_logp and exp(d)-1-d.
        # A deterministic shift makes d=2, hence KL estimate exp(2)-3 > .1.
        # Actor and reference remain identical, so reference KL stays near zero.
        trajectory['old_logprobs'] = trajectory['old_logprobs'] - 2.
        actor_before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        optimizer = torch.optim.AdamW([dict(params=policy.parameters(), lr=.001),
                                       dict(params=critic.parameters(), lr=.01)], weight_decay=0.)
        settings = dict(ppo_epochs=2, clip_range=.2, value_clip_range=.2, value_coefficient=.5,
                        entropy_coefficient=0., grad_clip=1., target_kl=.1, reference_kl_limit=100.)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            metrics = optimize_rollout(policy, reference, critic, optimizer, [trajectory], legal, settings, 'cpu')
            step.assert_not_called()
        self.assertTrue(metrics['epoch_kl_stop'])
        self.assertFalse(metrics['reference_stop'])
        self.assertEqual(metrics['ppo_epochs_completed'], 0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertAlmostEqual(metrics['old_policy_kl'], torch.exp(torch.tensor(2.)).item() - 3., places=4)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()


