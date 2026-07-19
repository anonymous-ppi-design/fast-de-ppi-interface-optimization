from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
@dataclass
class RFHConfig:
    n_heads: int = 8
    feature_dim: int = 256
    ridge_alpha: float = 1.0
    bootstrap_fraction: float = 0.8
    uncertainty_penalty: float = 0.05
    seed: int = 37
class RandomFrozenHeadEnsemble:
    def __init__(self, z_dim: int, config: RFHConfig, device='cpu'):
        self.z_dim=z_dim; self.config=config; self.device=device; self.heads=[]
    def _features(self,z,W,b): return F.gelu(z @ W.T + b)
    def fit(self,z_np,y_np,weights_np=None):
        rng=np.random.default_rng(self.config.seed); z=torch.tensor(z_np,dtype=torch.float32,device=self.device); y=torch.tensor(y_np,dtype=torch.float32,device=self.device); w=torch.ones_like(y) if weights_np is None else torch.tensor(weights_np,dtype=torch.float32,device=self.device); w=w.clamp(min=0.05); n=z.shape[0]; self.heads=[]
        for _ in range(self.config.n_heads):
            W=torch.tensor(rng.normal(0,1/np.sqrt(self.z_dim),size=(self.config.feature_dim,self.z_dim)),dtype=torch.float32,device=self.device); b=torch.tensor(rng.normal(0,0.5,size=(self.config.feature_dim,)),dtype=torch.float32,device=self.device)
            idx=torch.tensor(rng.choice(np.arange(n),size=max(8,int(self.config.bootstrap_fraction*n)),replace=True),dtype=torch.long,device=self.device); Phi=self._features(z[idx],W,b); yy=y[idx]; sw=torch.sqrt(w[idx]).unsqueeze(1); A=Phi*sw; lhs=A.T@A+self.config.ridge_alpha*torch.eye(self.config.feature_dim,device=self.device); rhs=A.T@(yy*sw.squeeze(1)); beta=torch.linalg.solve(lhs,rhs); self.heads.append({'W':W.detach(),'b':b.detach(),'beta':beta.detach()})
    def predict_torch(self,z):
        p=torch.stack([self._features(z,h['W'],h['b'])@h['beta'] for h in self.heads],dim=0); return p.mean(0), p.std(0)
    def objective_torch(self,z):
        m,s=self.predict_torch(z); return m - self.config.uncertainty_penalty*s
