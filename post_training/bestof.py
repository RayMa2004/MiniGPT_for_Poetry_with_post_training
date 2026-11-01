"""Reward-model best-of-K preparation for iterative rejection-sampling SFT."""
import argparse
import math
from collections import defaultdict, Counter
from .core import (SPLITS, resolve, load_config, read_complete_samples, read_json,
                   file_hash, digest, artifact_path, new_output, write_jsonl, atomic_json)
from .data import checked_samples
from .prepared import readiness


def select_candidates(samples, scores, top_k=1):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    checked_samples(samples)
    if set(scores) != {row['sample_id'] for row in samples}:
        raise ValueError("Reward scores must cover each sample exactly once")
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Reward scores must be finite numbers")
    # Exact texts shared across splits belong exclusively to the most held-out split.
    priority={'test':0,'val':1,'train':2}
    owner={}
    for row in samples:
        text=row['text']; split=row['split']
        if text not in owner or priority[split] < priority[owner[text]]:
            owner[text]=split
    groups=defaultdict(list)
    for row in samples:
        if owner[row['text']] == row['split']:
            groups[row['task_id']].append(row)
    selected={split:[] for split in SPLITS}
    seen=set()
    for task,rows in sorted(groups.items()):
        count=0
        for row in sorted(rows,key=lambda x:(-scores[x['sample_id']],x['sample_id'])):
            if row['text'] in seen:
                continue
            selected[row['split']].append(dict(row,reward_score=scores[row['sample_id']]))
            seen.add(row['text']); count+=1
            if count >= top_k: break
    return selected


def score_candidates(rows, reward, tokenizer, batch_size, device):
    import torch
    from .learning import full_batch, chunks
    scores={}
    reward.eval()
    with torch.no_grad():
        for batch in chunks(rows,batch_size):
            ids,lengths=full_batch(batch,tokenizer,reward.backbone.block_size,device)
            values=reward(ids,lengths).tolist()
            for row,value in zip(batch,values):
                if not math.isfinite(value): raise ValueError('Reward model produced nonfinite score')
                scores[row['sample_id']]=float(value)
    return scores


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',required=True)
    parser.add_argument('--reward-checkpoint',required=True)
    parser.add_argument('--reward-validation',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default='configs/post_training_rm_ppo_experiment.json')
    parser.add_argument('--top-k',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if args.batch_size < 1 or args.top_k < 1:
        parser.error('--batch-size and --top-k must be positive')
    target=artifact_path(args.output)
    if target.exists():
        raise ValueError(f'Output already exists: {target}; use a new output path')
    cfg=load_config(args.config)
    samples_path=resolve(args.samples)
    reward_path=resolve(args.reward_checkpoint)
    validation_path=resolve(args.reward_validation)
    vocab_path=resolve(cfg['paths']['vocab'])
    rows=checked_samples(read_complete_samples(samples_path))
    if not rows: raise ValueError('No candidate samples')
    vocab_hash=file_hash(vocab_path)
    if any(row['vocab_sha256'] != vocab_hash for row in rows):
        raise ValueError('Candidate vocabulary does not match reward vocabulary')
    from .ppo import validate_reward_gate
    report=read_json(validation_path)
    validate_reward_gate(report,reward_path,vocab_path,cfg)
    from tokenizer import CharTokenizer
    from .learning import load_reward
    tokenizer=CharTokenizer.load(vocab_path)
    reward,payload=load_reward(reward_path,tokenizer,vocab_path,args.device)
    validate_reward_gate(report,reward_path,vocab_path,cfg,payload)
    scores=score_candidates(rows,reward,tokenizer,args.batch_size,args.device)
    selected=select_candidates(rows,scores,args.top_k)
    ready=readiness(selected)
    print('readiness:',ready,flush=True)
    out=new_output(args.output)
    for split in SPLITS:
        write_jsonl(out/'sft'/f'{split}.jsonl',selected[split])
        write_jsonl(out/'preferences'/f'{split}.jsonl',[])
    write_jsonl(out/'reward_scores.jsonl',[{'sample_id':row['sample_id'], 'sample_hash':digest(row),
                 'reward_score':scores[row['sample_id']]} for row in rows])
    summary={'input_samples':len(rows),'sft':{split:{'count':len(values),
             'coverage':dict(Counter(f"{r['mode']}/{r['form']}" for r in values))}
             for split,values in selected.items()},'readiness':{'sft':ready}}
    outputs={f'{section}/{split}.jsonl':file_hash(out/section/f'{split}.jsonl')
             for section in ('sft','preferences') for split in SPLITS}
    outputs['reward_scores.jsonl']=file_hash(out/'reward_scores.jsonl')
    atomic_json(out/'manifest.json',{'kind':'reward_bestof_v1','inputs':{
        'samples':file_hash(samples_path),'candidate_manifest':file_hash(samples_path.parent/'manifest.json'),
        'reward_checkpoint':file_hash(reward_path),'reward_validation':file_hash(validation_path),
        'vocab':vocab_hash},'selection':{'top_k':args.top_k,'method':'reward_model_best_of_k',
        'tie_break':'sample_id','duplicates':'reserve most held-out split'},'summary':summary,'outputs':outputs})
    print('Prepared reward-selected data:',out,flush=True)
    if not ready['ready']:
        print('Data is not trainable: inspect missing groups; generate more candidates.',flush=True)
        return 2
    return 0
