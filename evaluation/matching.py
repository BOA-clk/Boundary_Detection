"""Protocol-explicit matching; no global claim that greedy is optimal."""
import numpy as np
from experiments import offline_main as reference

def bipartite_boundary_match(predictions,ground_truth,tolerance_sec):
    """Ablation max-cardinality/min-error FP64 assignment, with dummy columns."""
    return reference.mb_match(predictions,ground_truth,tolerance_sec)

def compute_boundary_prf(predictions,ground_truth,tolerance_sec,policy='legacy_greedy'):
    if policy=='legacy_greedy':
        p,r,f=reference.boundary_prf(predictions,ground_truth,tolerance_sec)
        tp=int(round(r*len(ground_truth)))
    elif policy=='optimal_assignment':
        tp=len(bipartite_boundary_match(predictions,ground_truth,tolerance_sec))
        p=tp/len(predictions) if len(predictions) else 0.
        r=tp/len(ground_truth) if len(ground_truth) else 0.
        f=2*tp/(len(predictions)+len(ground_truth)) if len(predictions)+len(ground_truth) else 0.
    elif policy=='streaming_absolute_cost':
        from scipy.optimize import linear_sum_assignment
        pred,gt=np.asarray(predictions),np.asarray(ground_truth)
        cost=np.abs(gt[:,None]-pred[None,:])
        rows,cols=linear_sum_assignment(cost)
        tp=int(np.sum(cost[rows,cols]<=tolerance_sec))
        p=tp/len(pred) if len(pred) else 0.; r=tp/len(gt) if len(gt) else 0.
        f=2*p*r/(p+r) if p+r else 0.
    else: raise ValueError(policy)
    return dict(precision=p,recall=r,f1=f,tp=tp,n_pred=len(predictions),n_gt=len(ground_truth))

def evaluate_dataset_macro_f1(predictions_per_video,ground_truth_per_video,tolerances,policy='legacy_greedy'):
    if set(predictions_per_video)!=set(ground_truth_per_video):
        raise ValueError('Prediction/GT video IDs differ; missing videos must not be silently dropped')
    if not predictions_per_video: raise ValueError('No test videos')
    result={}
    for tol in tolerances:
        values=[compute_boundary_prf(predictions_per_video[v],ground_truth_per_video[v],tol,policy)
                for v in predictions_per_video]
        result[tol]={k:float(np.mean([row[k] for row in values])) for k in ('precision','recall','f1')}
    return result
