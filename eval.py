"""Offline evaluation with saved validation calibration, never test tuning.

Only load trusted checkpoints produced by this repository: torch.load uses pickle.
Missing data/checkpoints fail instead of using random weights or synthetic samples.
"""
import argparse
import json
from pathlib import Path
from data.dataset import DATASET_SPECS,load_protocol_data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=('main','ablation'),default='main')
    p.add_argument('--dataset',choices=DATASET_SPECS,required=True)
    p.add_argument('--data_root',required=True)
    p.add_argument('--run_dir',required=True,help='Contains manifest.json, best.pt and result.json')
    args=p.parse_args(); run=Path(args.run_dir)
    manifest=json.loads((run/'manifest.json').read_text())
    result=json.loads((run/'result.json').read_text())
    if manifest['spec']['name']!=args.dataset: p.error('Dataset/checkpoint mismatch')
    if args.protocol=='main' and result.get('primary_metric_policy')!='legacy_greedy': p.error('Not a main legacy_greedy run')
    if args.protocol=='ablation' and 'primary_metric_policy' in result: p.error('Main checkpoint supplied for ablation')
    runner,_,videos=load_protocol_data(args.dataset,args.data_root,args.protocol)
    saved_signatures=manifest.get('signatures',manifest.get('timing',{}))
    for vid in manifest['ids']['test']:
        saved=saved_signatures[vid]
        current=videos[vid].signature
        for key in ('labels_sha256','shape','native_hz','stride','n_steps'):
            if saved[key]!=current[key]:
                raise ValueError(f'Data no longer matches checkpoint manifest: {vid}/{key}')
    spec=runner.MB_DATASETS[args.dataset]; config=runner.MBTrainConfig(**manifest['config'])
    device=runner.mb_runtime_preflight(config)
    model=runner.MamboaV2(spec,config,'full') if args.protocol=='main' else runner.MBAblModel(
        spec,config,variant=run.parents[2].name)
    model=model.to(device)
    state=runner.mb_load_own_checkpoint(run/'best.pt')
    if state['fingerprint']!=result['fingerprint']: raise ValueError('Checkpoint/result mismatch')
    model.load_state_dict(state['model'])
    logits,_,_=runner.mb_predict(model,manifest['ids']['test'],videos,config,device)
    cal=result['calibration']
    if args.protocol=='main':
        pred=runner.m5_decode_logits(logits,videos,spec,config,cal['lam'])
        pred=runner.m5_subtract_offset(pred,cal['offset_sec'])
        report=runner.m5_score(pred,videos,spec,'legacy_greedy')
    else: report=runner.mb_evaluate_logits(logits,videos,spec,config,cal['lam'],cal['offset_sec'])
    runner.mb_write_json(run/'evaluation_recomputed.json',report)
    print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__': main()
