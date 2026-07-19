from __future__ import annotations
import numpy as np
import torch
from fastde.data.sequence_utils import AMINO_ACIDS, IDX_TO_AA
def encode_sequences(model, token_tensor, batch_size=256, device='cpu'):
    model.eval(); outs=[]
    with torch.no_grad():
        for start in range(0,len(token_tensor),batch_size):
            x=token_tensor[start:start+batch_size].to(device); mu,_=model.encode(x); outs.append(mu.detach().cpu().numpy())
    return np.vstack(outs)
def decode_latents(model,z,temperature=0.9,mutation_budget_min=3,mutation_budget_max=25):
    model.eval()
    with torch.no_grad():
        aa_logits,gate_logits=model.decode_logits(z); aa_probs=torch.softmax(aa_logits/temperature,dim=-1).detach().cpu().numpy(); gate=torch.sigmoid(gate_logits).detach().cpu().numpy()
    wt=model.wt_tokens.detach().cpu().numpy(); seqs=[]
    for b in range(z.shape[0]):
        k=np.random.randint(mutation_budget_min,mutation_budget_max+1); positions=np.argsort(-gate[b])[:k]; arr=wt.copy()
        for pos in positions:
            p=aa_probs[b,pos].copy(); p[wt[pos]]=0.0
            if p.sum()<=1e-12: p[:]=1.0; p[wt[pos]]=0.0
            p=p/p.sum(); arr[pos]=np.random.choice(np.arange(len(AMINO_ACIDS)),p=p)
        seqs.append(''.join(IDX_TO_AA[int(i)] for i in arr))
    return seqs
def optimize_latents(z0,objective_fn,steps=20,lr=0.05,l2_penalty=0.01,device='cpu'):
    z=torch.tensor(z0,dtype=torch.float32,device=device,requires_grad=True); z_start=z.detach().clone(); opt=torch.optim.Adam([z],lr=lr)
    for _ in range(steps):
        score=objective_fn(z); l2=((z-z_start)**2).mean(dim=1); loss=-(score-l2_penalty*l2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return z.detach().cpu().numpy()
