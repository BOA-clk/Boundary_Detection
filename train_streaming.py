"""Separate TCN-free streaming reference training/evaluation.

This entry point is not the Adam/50 main-table runner. Historical fold-specific
streaming thresholds and environment choices were not recovered; current options
are recorded explicitly and must not be relabelled as recovered historical settings.
"""
import argparse,json,math,random
from pathlib import Path
import numpy as np
import torch
from data.dataset import DATASET_SPECS,load_dataset_split
from models.streaming_pipeline import StreamingBoundaryModel,StreamingDetector
from models.full_pipeline import compute_gaussian_target,weighted_bce_loss
from models.groupwise_mamba import MAMBA_AVAILABLE,FallbackScan
from decoding.dp_decoder import decode_boundaries_dp
from evaluation.matching import evaluate_dataset_macro_f1

def score(model,samples,spec,device,streaming=False,threshold=.5):
    predictions={}; truth={}
    with torch.no_grad():
        model.eval()
        for v in samples:
            features=v.features.to(device)
            if streaming:
                detector=StreamingDetector(model,feature_hz=15.,peak_threshold=threshold,
                    min_gap_sec=1.,device=device)
                pred=[]
                for frame in features:
                    event=detector.push_feature(frame)
                    if event['is_confirmed_boundary']: pred.append(event['confirmed_boundary_sec'])
            else:
                logits=model(features)['logits'].float().cpu().numpy()
                pred=decode_boundaries_dp(logits,v.times,v.duration_sec,penalty_lambda=1.,
                    min_gap_sec=spec.min_gap_sec,feature_hz=15.)
            predictions[v.video_id]=pred;truth[v.video_id]=v.ground_truth_boundaries
    return evaluate_dataset_macro_f1(predictions,truth,spec.tolerances,policy='streaming_absolute_cost')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('train','eval'))
    p.add_argument('--dataset',choices=DATASET_SPECS,default='50salads')
    p.add_argument('--data_root',required=True)
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--groups',type=int,choices=(1,32,64),default=32)
    p.add_argument('--checkpoint',default='outputs/streaming/best.pt')
    p.add_argument('--peak_threshold',type=float,default=.5)
    args=p.parse_args()
    if not torch.cuda.is_available() or not MAMBA_AVAILABLE:
        p.error('Streaming training/evaluation requires CUDA mamba_ssm; use demo_streaming.py for CPU smoke tests')
    if not 1<=args.fold<=DATASET_SPECS[args.dataset].folds:p.error('Invalid fold')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda');spec=DATASET_SPECS[args.dataset]
    model=StreamingBoundaryModel(n_groups=args.groups).to(device)
    if any(isinstance(x,FallbackScan) for x in model.modules()):raise RuntimeError('Fallback is not a paper backend')
    # Main-style seed-dependent held-out split; stored below for auditability.
    load=lambda kind:load_dataset_split(args.dataset,args.data_root,args.fold,kind,args.seed,'main')
    checkpoint=Path(args.checkpoint)
    if args.action=='eval':
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        for k in ('dataset','fold','seed','groups'):
            if state['settings'][k]!=getattr(args,k):raise ValueError('Checkpoint/settings mismatch: '+k)
        model.load_state_dict(state['model'])
        result=score(model,load('test'),spec,device,True,args.peak_threshold)
        print(json.dumps(result,indent=2));return
    if checkpoint.exists():raise FileExistsError('Choose a new checkpoint path')
    train,val=load('train'),load('val')
    mass=sum(float(compute_gaussian_target(v.times,v.ground_truth_boundaries,spec.sigma_sec).sum()) for v in train)
    model.init_prior(mass/sum(v.n_frames for v in train))
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.01)
    steps=60*len(train);warm=max(1,round(.1*steps))
    def lr_scale(step):
        if step<warm:return (step+1)/warm
        progress=min(1.,(step-warm)/max(1,steps-warm))
        return (1e-6+(5e-4-1e-6)*.5*(1+math.cos(math.pi*progress)))/5e-4
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_scale)
    checkpoint.parent.mkdir(parents=True,exist_ok=True);best=-1.;history=[]
    for epoch in range(60):
        model.train();losses=[]
        for i in np.random.RandomState(args.seed+epoch).permutation(len(train)):
            v=train[i];features=v.features.to(device)
            target=compute_gaussian_target(v.times,v.ground_truth_boundaries,spec.sigma_sec).to(device)
            opt.zero_grad(set_to_none=True);loss=weighted_bce_loss(model(features)['logits'],target)
            if not torch.isfinite(loss):raise FloatingPointError(v.video_id)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();scheduler.step();losses.append(float(loss.detach()))
        f1=score(model,val,spec,device)[spec.tolerances[1]]['f1']
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_f1=f1)
        history.append(row);print(row,flush=True)
        if f1>best:
            best=f1;torch.save(dict(model=model.state_dict(),epoch=epoch+1,settings=vars(args),
                precision='fp32',split_seed=args.seed,validation_ids=[v.video_id for v in val],
                train_ids=[v.video_id for v in train]),checkpoint)
        checkpoint.with_suffix('.history.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__': main()
