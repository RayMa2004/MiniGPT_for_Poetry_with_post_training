"""Remote-only rejection-sampling fine-tuning with continuation-only targets."""
import argparse
import math
import random
from .core import resolve, load_config, file_hash, digest


def validation_loss(model, rows, tokenizer, batch_size, device):
    import torch
    from .learning import chunks, supervised_batch
    model.eval()
    total,tokens=0.,0
    with torch.no_grad():
        for batch in chunks(rows,batch_size):
            x,y=supervised_batch(batch,tokenizer,model.block_size,device)
            count=int((y != -100).sum().item())
            _,loss=model(x,y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SFT validation loss")
            total+=loss.item()*count
            tokens+=count
    if not tokens:
        raise ValueError("Validation has no continuation targets")
    return total/tokens


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,help="Prepared dataset directory")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--output",required=True)
    parser.add_argument("--config")
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args(argv)
    config=load_config(args.config)
    from .prepared import load_prepared
    data,manifest=load_prepared(args.data,"sft")
    from .learning import (seed_all, validate_hyperparameters, load_policy,
                           TrainingRun, chunks, supervised_batch, export_policy, encode_sequence)
    import torch
    validate_hyperparameters(config,"sft")
    seed_all(config["seed"])
    checkpoint=resolve(args.init_checkpoint or config["paths"]["checkpoint"])
    vocab=resolve(config["paths"]["vocab"])
    origin=file_hash(checkpoint)
    vocabulary=file_hash(vocab)
    for rows in data.values():
        if any(row["checkpoint_sha256"] != origin or row["vocab_sha256"] != vocabulary for row in rows):
            raise ValueError("RSFT候选必须来自指定初始化模型和同一词表；新来源应重新采样")
    model,tokenizer,model_config=load_policy(checkpoint,vocab,args.device)
    for rows in data.values():
        for row in rows:
            encode_sequence(row,tokenizer,model.block_size)
    t=config["sft"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
    inputs={"checkpoint":checkpoint,"vocab":vocab,"prepared_manifest":resolve(args.data)/"manifest.json"}
    with TrainingRun(args.output,"sft",config,inputs) as run:
        best=math.inf
        step=0
        for epoch in range(1,t["epochs"]+1):
            rows=list(data["train"])
            random.Random(config["seed"]+epoch).shuffle(rows)
            model.train()
            loss_sum,tokens=0.,0
            for batch in chunks(rows,t["batch_size"]):
                x,y=supervised_batch(batch,tokenizer,model.block_size,args.device)
                optimizer.zero_grad(set_to_none=True)
                _,loss=model(x,y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite SFT loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),t["grad_clip"],error_if_nonfinite=True)
                optimizer.step()
                count=int((y != -100).sum().item())
                loss_sum+=loss.item()*count
                tokens+=count
                step+=1
            val=validation_loss(model,data["val"],tokenizer,t["batch_size"],args.device)
            run.log({"epoch":epoch,"step":step,"train_continuation_loss":loss_sum/tokens,"val_continuation_loss":val})
            metadata={"stage":"rsft","epoch":epoch,"step":step,"source_checkpoint_sha256":origin,
                      "prepared_manifest_sha256":digest(manifest),"val_continuation_loss":val,
                      "selection_metric":"val_continuation_loss; not a poetry-quality guarantee"}
            export_policy(run.path/"last.pt",model,model_config,vocab,metadata)
            if val < best:
                best=val
                export_policy(run.path/"best.pt",model,model_config,vocab,metadata)
            print(f"RSFT epoch={epoch} train={loss_sum/tokens:.4f} val={val:.4f}",flush=True)
        run.metadata.update(best_val_continuation_loss=best,steps=step,
                            quality_status="Needs held-out blinded quality evaluation")


if __name__ == "__main__":
    main()
