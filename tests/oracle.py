"""CPU-only independent parameter-shape/recurrence oracle; never an experiment backend."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MambaDouble(nn.Module):
    """Independent recurrence oracle, not a replacement for production Mamba."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner, self.dt_rank = d_model*expand, math.ceil(d_model/16)
        e, r = self.d_inner, self.dt_rank
        self.in_proj = nn.Linear(d_model, 2*e, bias=False)
        self.conv1d = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv-1)
        self.x_proj = nn.Linear(e, r+2*d_state, bias=False)
        self.dt_proj = nn.Linear(r, e)
        self.A_log = nn.Parameter(torch.randn(e, d_state)*.1)
        self.D = nn.Parameter(torch.ones(e))
        self.A_log._no_weight_decay = self.D._no_weight_decay = True
        self.out_proj = nn.Linear(e, d_model, bias=False)
        self.act, self.activation = nn.SiLU(), 'silu'

    def allocate_inference_cache(self, batch_size, max_seqlen):
        return (torch.zeros(batch_size, self.d_inner, self.d_conv),
                torch.zeros(batch_size, self.d_inner, self.d_state))

    def step(self, frame, conv, state):
        x, gate = self.in_proj(frame[:, 0]).chunk(2, -1)
        conv = torch.cat((conv[:, :, 1:], x.unsqueeze(-1)), -1)
        v = F.silu((conv*self.conv1d.weight[:, 0]).sum(-1)+self.conv1d.bias)
        low, b, c = self.x_proj(v).split((self.dt_rank,self.d_state,self.d_state),-1)
        delta = F.softplus(self.dt_proj(low))
        decay = torch.exp(-delta.unsqueeze(-1)*self.A_log.exp())
        state = decay*state + delta.unsqueeze(-1)*v.unsqueeze(-1)*b.unsqueeze(1)
        out = ((state*c.unsqueeze(1)).sum(-1)+self.D*v)*F.silu(gate)
        return self.out_proj(out).unsqueeze(1), conv, state

    def forward(self, x):
        conv, state = self.allocate_inference_cache(x.shape[0], x.shape[1])
        outputs = []
        for t in range(x.shape[1]):
            y, conv, state = self.step(x[:, t:t+1],conv,state)
            outputs.append(y)
        return torch.cat(outputs,1)
