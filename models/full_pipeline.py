"""Offline model API backed by the actual main-table runner.

Uses mamba_ssm.Mamba only. The frozen unused direct head is retained so
parameter counts and checkpoint keys agree with the experiment code.
"""
from dataclasses import replace
import torch
from experiments import offline_main as reference
DATASET_DILATIONS={'breakfast':(1,11,23),'gtea':(1,7,13),'50salads':(1,17,35)}
DATASET_SIGMA_SEC={k:v.sigma_sec for k,v in reference.MB_DATASETS.items()}
ResidualDilatedTCN=reference.MBTemporalContext

class FullBoundaryModel(reference.MamboaV2):
    def __init__(self,dataset='50salads',input_dim=2048,n_groups=32,context_dim=64,dilations=None):
        config=replace(reference.M5_CONFIG,input_dim=input_dim,n_groups=n_groups,
                       context_dim=context_dim,context_dilations=dilations)
        super().__init__(reference.MB_DATASETS[dataset],config,'full')
    def forward(self,features):
        output=super().forward(features)
        return dict(output,logits=output['logit'],probs=torch.sigmoid(output['logit']))

def compute_gaussian_target(times,ground_truth_boundaries,sigma_sec=1.):
    return torch.from_numpy(reference.mb_gaussian_target(times,ground_truth_boundaries,sigma_sec))

def weighted_bce_loss(logits,targets,pos_weight_alpha=4.):
    weights=1.+pos_weight_alpha*targets
    bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return (weights*bce).sum()/weights.sum()
