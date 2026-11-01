"""Resumable paired and absolute AI evaluation for the completed second round."""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from post_training import compare, data, judge
from post_training.core import (resolve, read_json, read_jsonl, read_complete_samples, digest,
                                file_hash, atomic_json, write_jsonl, new_output, artifact_path, load_config)
from post_training.rubric import DIMENSIONS, validate_result

SOURCES={
    'baseline':'post_training_data/fresh1_candidates/samples.jsonl',
    'ppo1':'post_training_data/ppo_v1_candidates/samples.jsonl',
    'round2_rsft':'post_training_data/round2_rsft_eval/samples.jsonl',
    'round2_ppo':'post_training_data/round2_ppo_eval/samples.jsonl',
}


def ensure_jsonl(path, rows):
    path=Path(path)
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(f'Existing input differs: {path}; preserve it and choose a new output root')
    else:
        write_jsonl(path,rows)


def ensure_json(path, value):
    path=Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing artifact differs: {path}; do not overwrite it')
    else:
        atomic_json(path,value)


def quality_summary(samples, labels, settings):
    by_id=data.checked_labels(samples,labels,'sample_id')
    if len(by_id) != len(samples):
        raise ValueError(f'Incomplete quality labels: {len(by_id)}/{len(samples)}')
    grouped=defaultdict(list)
    for row in samples:
        result=by_id[row['sample_id']]['result']
        validate_result('score',result)
        grouped[f"{row['mode']}/{row['form']}"].append(result['scores'])
    def summarize(scores):
        if not scores:
            raise ValueError('Quality summary has no scores')
        qualified=sum(sum(s.values())/len(DIMENSIONS) >= settings['minimum_mean'] and
                      min(s.values()) >= settings['minimum_dimension'] for s in scores)
        return {'count':len(scores), 'dimension_mean':{d:sum(s[d] for s in scores)/len(scores) for d in DIMENSIONS},
                'qualified_count':qualified, 'qualified_rate':qualified/len(scores)}
    flat=[s for values in grouped.values() for s in values]
    output=summarize(flat)
    output['by_group']={key:summarize(values) for key,values in sorted(grouped.items())}
    return output


def prepare_comparison(out, baseline, candidate, split, left, right):
    expected=compare.comparison_pairs(left,right,split=split)
    path=out/'comparison.json'
    if out.exists():
        if not path.is_file() or not (out/'pairs.jsonl').is_file() or not (out/'manifest.json').is_file():
            raise ValueError(f'Incomplete comparison: {out}; preserve it and use a new output root')
        manifest=read_json(out/'manifest.json')
        if read_json(path) != expected or read_jsonl(out/'pairs.jsonl') != expected['pairs']:
            raise ValueError(f'Existing comparison content differs: {out}')
        wanted={'baseline':file_hash(resolve(baseline)),'candidate':file_hash(resolve(candidate)),
                'comparison_sha256':file_hash(path)}
        if manifest != wanted:
            raise ValueError(f'Existing comparison identity differs: {out}')
    else:
        new_output(out)
        write_jsonl(out/'pairs.jsonl',expected['pairs'])
        atomic_json(path,expected)
        atomic_json(out/'manifest.json',{'baseline':file_hash(resolve(baseline)),
            'candidate':file_hash(resolve(candidate)), 'comparison_sha256':file_hash(path)})
    return expected


def run_labels(config, source, destination, kind):
    summary=judge.run_judging(config,source,destination,kind,execute=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if summary['remaining_items'] or summary['failed_items']:
        print('标注未全部完成；检查错误文件并重新运行同一命令，已成功结果会复用。',flush=True)
        return False
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('val','test'),required=True)
    parser.add_argument('--config',default='configs/post_training_v3.json')
    parser.add_argument('--execute',action='store_true',help='Explicitly make AI API requests')
    parser.add_argument('--output',help='New/reusable directory under post_training_data or post_training_runs')
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    source_rows={name:read_complete_samples(resolve(path)) for name,path in SOURCES.items()}
    cases=[('base_vs_round2_rsft','baseline','round2_rsft'),
           ('base_vs_round2_ppo','baseline','round2_ppo')]
    if args.split=='val':
        cases.append(('ppo1_vs_round2_rsft','ppo1','round2_rsft'))
    else:
        cases.append(('round2_rsft_vs_ppo','round2_rsft','round2_ppo'))
    comparisons={name:compare.comparison_pairs(source_rows[a],source_rows[b],split=args.split)
                 for name,a,b in cases}
    models=('baseline','round2_rsft','round2_ppo')
    score_inputs={name:[r for r in source_rows[name] if r['split']==args.split] for name in models}
    out=artifact_path(args.output or f'post_training_data/round2_full_eval_{args.split}')
    print('评估分区:',args.split,'输出目录:',out,flush=True)
    for name,_,_ in cases:
        result=comparisons[name]
        print(f"{name}: matched={result['matched_count']} identical={len(result['identical'])} pairs={len(result['pairs'])}",flush=True)
    print('单首四维评分样本数:',{name:len(rows) for name,rows in score_inputs.items()},flush=True)
    if not args.execute:
        print('仅查看计划；未创建文件、未读取密钥、未发送 API 请求。')
        return 0
    key=os.environ.get('POSTTRAIN_API_KEY')
    if not key or any(ch.isspace() for ch in key):
        raise ValueError('POSTTRAIN_API_KEY 未设置或包含空白；请在本终端安全输入')
    out.mkdir(parents=True,exist_ok=True)
    for name,a,b in cases:
        print(f"\n配对盲评：{name}",flush=True)
        directory=out/name
        comparison=prepare_comparison(directory,SOURCES[a],SOURCES[b],args.split,source_rows[a],source_rows[b])
        pair_file=directory/'pairs.jsonl'; labels_file=directory/'preferences.jsonl'
        if comparison['pairs'] and not run_labels(cfg,pair_file,labels_file,'pairs'):
            return 1
        labels=read_jsonl(labels_file) if labels_file.exists() else []
        result=compare.summarize_comparison(comparison,labels)
        ensure_json(directory/'report.json',result)
        print('配对报告:',directory/'report.json',flush=True)
    reports={}
    judge_ids=set()
    for name,rows in score_inputs.items():
        print(f"\n逐首四维评分：{name}",flush=True)
        input_path=out/f'{name}_samples.jsonl'
        output_path=out/f'{name}_scores.jsonl'
        ensure_jsonl(input_path,rows)
        if not run_labels(cfg,input_path,output_path,'score'):
            return 1
        labels=read_jsonl(output_path)
        judge_ids.update(r['judge_id'] for r in labels)
        reports[name]=quality_summary(rows,labels,cfg['selection'])
    if len(judge_ids)!=1:
        raise ValueError('不同版本逐首评分的评审配置不一致，不能直接比较')
    summary={'split':args.split,'scope':'AI per-poem ratings; held-out task groups, not human ground truth',
             'quality_threshold':{'minimum_mean':cfg['selection']['minimum_mean'],
                                  'minimum_dimension':cfg['selection']['minimum_dimension']},
             'judge_id':next(iter(judge_ids)),'models':reports}
    ensure_json(out/'quality_summary.json',summary)
    print('四维质量报告:',out/'quality_summary.json',flush=True)
    print('评估完成。',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError) as error:
        # Never echo credentials if a local validation error contains one.
        message=str(error)
        key=os.environ.get('POSTTRAIN_API_KEY')
        if key: message=message.replace(key,'[REDACTED]')
        print(f'round2_eval: {message}',file=sys.stderr)
        raise SystemExit(2)
