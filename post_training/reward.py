"""Remote-only preference reward learning; validation gate is identity-bound."""
import argparse
import math
import random
from collections import defaultdict
from .core import resolve, load_config, file_hash, atomic_json, digest


def evaluate_reward(reward, rows, tokenizer, batch_size, device):
    import torch
    from .learning import chunks, full_batch, ranking_loss
    correct=0
    loss_sum=0.
    rewards=[]
    groups=defaultdict(lambda:{"count":0,"correct":0})
    reward.eval()
    with torch.no_grad():
        for batch in chunks(rows,batch_size):
            a,alen=full_batch([r["chosen"] for r in batch],tokenizer,reward.backbone.block_size,device)
            b,blen=full_batch([r["rejected"] for r in batch],tokenizer,reward.backbone.block_size,device)
            chosen,rejected=reward(a,alen),reward(b,blen)
            if not torch.isfinite(chosen).all() or not torch.isfinite(rejected).all():
                raise RuntimeError("Nonfinite reward validation scores")
            wins=(chosen > rejected).tolist()
            correct+=sum(wins)
            loss_sum+=ranking_loss(chosen,rejected).item()*len(batch)
            rewards.extend(chosen.tolist()+rejected.tolist())
            for row,win in zip(batch,wins):
                key=row["chosen"]["mode"]+"/"+row["chosen"]["form"]
                groups[key]["count"]+=1
                groups[key]["correct"]+=int(win)
    count=len(rows)
    mean=sum(rewards)/len(rewards)
    std=(sum((v-mean)**2 for v in rewards)/len(rewards))**0.5
    return {"val_pair_count":count,"val_accuracy":correct/count,"val_pair_loss":loss_sum/count,
            "by_group":{key:{"count":v["count"],"accuracy":v["correct"]/v["count"]} for key,v in groups.items()},
            "normalization":{"mean":mean,"std":std}}


def gate_passes(metrics, settings):
    expected={m+"/"+f for m in ("prefix","free") for f in ("wujue","qijue")}
    groups=metrics["by_group"]
    return (metrics["val_pair_count"] >= settings["minimum_val_pairs"]
            and metrics["val_accuracy"] >= settings["minimum_val_accuracy"]
            and set(groups) == expected
            and all(v["count"] > 0 and v["accuracy"] > 0.5 for v in groups.values())
            and math.isfinite(metrics["normalization"]["std"]) and metrics["normalization"]["std"] > 1e-6)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--init-checkpoint",required=True,help="Explicit RSFT/baseline backbone initialization")
    parser.add_argument("--output",required=True)
    parser.add_argument("--config")
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args(argv)
    config=load_config(args.config)
    from .learning import (seed_all, load_prepared, validate_hyperparameters, load_policy, RewardModel,
                           TrainingRun, chunks, full_batch, ranking_loss)
    from checkpoint import save_checkpoint, FORMAT_VERSION
    import torch
    validate_hyperparameters(config,"reward")
    seed_all(config["seed"])
    t=config["reward"]
    if type(t["minimum_val_pairs"]) is not int or t["minimum_val_pairs"] < 1 or not .5 < t["minimum_val_accuracy"] <= 1:
        raise ValueError("Reward validation thresholds must require evidence above chance")
    data,manifest=load_prepared(args.data,"preferences")
    checkpoint=resolve(args.init_checkpoint)
    vocab=resolve(config["paths"]["vocab"])
    vocabulary=file_hash(vocab)
    if any(sample["vocab_sha256"] != vocabulary for rows in data.values() for row in rows for sample in (row["chosen"],row["rejected"])):
        raise ValueError("Preference samples have incompatible vocabulary")
    backbone,tokenizer,model_config=load_policy(checkpoint,vocab,args.device)
    unfrozen=t["unfreeze_last_blocks"]
    if type(unfrozen) is not int or not 0 <= unfrozen <= len(backbone.blocks):
        raise ValueError("unfreeze_last_blocks outside backbone range")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if unfrozen:
        for block in list(backbone.blocks)[-unfrozen:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in backbone.ln_f.parameters():
            parameter.requires_grad_(True)
    reward=RewardModel(backbone,model_config["n_embd"]).to(args.device)
    # Fixed dropout-free features, with gradients still enabled for explicitly trainable weights.
    reward.eval()
    optimizer=torch.optim.AdamW([p for p in reward.parameters() if p.requires_grad],lr=t["learning_rate"],weight_decay=t["weight_decay"])
    inputs={"checkpoint":checkpoint,"vocab":vocab,"prepared_manifest":resolve(args.data)/"manifest.json"}
    with TrainingRun(args.output,"reward",config,inputs) as run:
        best_key=(-math.inf,-math.inf)
        for epoch in range(1,t["epochs"]+1):
            rows=list(data["train"])
            random.Random(config["seed"]+epoch).shuffle(rows)
            loss_sum=0.
            for batch in chunks(rows,t["batch_size"]):
                a,alen=full_batch([r["chosen"] for r in batch],tokenizer,backbone.block_size,args.device)
                b,blen=full_batch([r["rejected"] for r in batch],tokenizer,backbone.block_size,args.device)
                optimizer.zero_grad(set_to_none=True)
                loss=ranking_loss(reward(a,alen),reward(b,blen))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite reward training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in reward.parameters() if p.requires_grad],t["grad_clip"],error_if_nonfinite=True)
                optimizer.step()
                loss_sum+=loss.item()*len(batch)
            metrics=evaluate_reward(reward,data["val"],tokenizer,t["batch_size"],args.device)
            run.log({"epoch":epoch,"train_pair_loss":loss_sum/len(rows),"val_pair_loss":metrics["val_pair_loss"],"val_accuracy":metrics["val_accuracy"]})
            payload={"format_version":FORMAT_VERSION,"kind":"post_training_reward_v1","model_config":model_config,
                     "backbone_state_dict":backbone.state_dict(),"reward_head_state_dict":reward.reward_head.state_dict(),
                     "vocab_sha256":vocabulary,"source_checkpoint_sha256":file_hash(checkpoint),
                     "prepared_manifest_sha256":digest(manifest),"reward_normalization":metrics["normalization"],
                     "validation":metrics,"epoch":epoch,"trainable_last_blocks":unfrozen}
            save_checkpoint(run.path/"last.pt",payload)
            key=(metrics["val_accuracy"],-metrics["val_pair_loss"])
            if key > best_key:
                best_key=key
                save_checkpoint(run.path/"best.pt",payload)
                report=dict(metrics,kind="reward_validation_v1",reward_checkpoint_sha256=file_hash(run.path/"best.pt"),
                            vocab_sha256=vocabulary,passed=gate_passes(metrics,t),
                            thresholds={k:t[k] for k in ("minimum_val_accuracy","minimum_val_pairs")},
                            training_source_sha256=digest(manifest),epoch=epoch)
                atomic_json(run.path/"reward_validation.json",report)
            print(f"Reward epoch={epoch} val_accuracy={metrics['val_accuracy']:.3f}",flush=True)
        run.metadata.update(best_val_accuracy=best_key[0],
                            next_step="PPO requires the exact best.pt and its passing reward_validation.json; otherwise collect/review more preferences.")


if __name__ == "__main__":
    main()
