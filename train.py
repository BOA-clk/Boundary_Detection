"""Paper offline training: main Adam/50 or ablation AdamW/60.

Requires real data and CUDA mamba_ssm.Mamba. No synthetic or CPU substitution.
"""
import argparse
from dataclasses import replace
from pathlib import Path
from data.dataset import DATASET_SPECS,load_protocol_data

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=('main','ablation'),default='main')
    p.add_argument('--dataset',choices=DATASET_SPECS,default='50salads')
    p.add_argument('--data_root',required=True)
    p.add_argument('--output_dir',default='./outputs')
    p.add_argument('--fold',type=int,help='Omit for all official folds')
    p.add_argument('--seed',type=int,choices=(0,1),help='Omit for both seeds')
    p.add_argument('--variant',default='full')
    p.add_argument('--mode',choices=('full',),default='full')
    args=p.parse_args()
    if args.protocol=='ablation' and args.dataset!='50salads': p.error('Ablations are 50Salads only')
    if args.protocol=='main' and args.variant!='full': p.error('Main uses the full model only')
    if args.fold is not None and not 1<=args.fold<=DATASET_SPECS[args.dataset].folds: p.error('Invalid fold')
    runner,by_seed,videos=load_protocol_data(args.dataset,args.data_root,args.protocol)
    if args.protocol=='ablation' and args.variant not in runner.MB_ABL_DESCRIPTIONS:
        p.error('Variants: '+', '.join(runner.MB_ABL_DESCRIPTIONS))
    spec=runner.MB_DATASETS[args.dataset]
    base=runner.M5_CONFIG if args.protocol=='main' else runner.MB_ABL_CONFIG
    if args.protocol=='main': runner.m5_guard()
    device=runner.mb_runtime_preflight(base)
    if args.protocol=='main':
        runner.m5_preflight(base,'legacy_greedy',device)
        runner.m5_longest_shape_check(spec,base,videos,device)
    else: runner.mba_check(args.variant,spec,runner.mba_config(args.variant,0),device)
    seeds=(args.seed,) if args.seed is not None else (0,1)
    folds=(args.fold,) if args.fold is not None else range(1,spec.folds+1)
    out=Path(args.output_dir)/args.protocol; rows=[]
    for seed in seeds:
        config=replace(base,seed=seed) if args.protocol=='main' else runner.mba_config(args.variant,seed)
        for fold in folds:
            ids=by_seed[seed][fold]
            if args.protocol=='main':
                run=out/'full'/spec.name/f'seed{seed}_fold{fold}'
                trainer=lambda:runner.m5_train_fold(spec,config,fold,ids,videos,out,device,'legacy_greedy')
            else:
                root=out/args.variant; run=root/'full'/spec.name/f'seed{seed}_fold{fold}'
                trainer=lambda:runner.mba_trainer(args.variant)(spec,config,fold,'full',ids,videos,root,device)
            run.mkdir(parents=True,exist_ok=True)
            from contextlib import redirect_stdout,redirect_stderr
            import sys,traceback
            class Tee:
                def __init__(self,console,log): self.console,self.log=console,log
                def write(self,v): self.console.write(v); self.log.write(v); return len(v)
                def flush(self): self.console.flush(); self.log.flush()
            with (run/'raw.log').open('a',buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                    try: row=trainer()
                    except BaseException: traceback.print_exc(); raise
            rows.append(row)
    target=out/('full' if args.protocol=='main' else args.variant)/spec.name
    runner.mb_write_json(target/'selected_run_results.json',rows)
    if args.seed is None and args.fold is None:
        summary=runner.m5_summary(rows,spec) if args.protocol=='main' else runner.mb_summary(rows,spec)
        runner.mb_write_json(target/'summary.json',summary)

if __name__=='__main__': main()
