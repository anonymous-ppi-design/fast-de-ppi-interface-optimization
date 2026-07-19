import torch.nn as nn
class TrainablePerformanceHead(nn.Module):
    def __init__(self, z_dim:int, hidden_dim:int=128, dropout:float=0.10):
        super().__init__(); self.net=nn.Sequential(nn.Linear(z_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,hidden_dim//2),nn.GELU(),nn.Linear(hidden_dim//2,1))
    def forward(self,z): return self.net(z).squeeze(-1)
