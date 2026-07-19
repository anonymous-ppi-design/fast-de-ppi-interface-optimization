from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastde.data.sequence_utils import AMINO_ACIDS, AA_TO_IDX, MASK_IDX
@dataclass
class VAEConfig:
    length: int
    latent_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.10
    kl_beta: float = 1e-4
    mutation_gate_weight: float = 0.25
class SparseMutationVAE(nn.Module):
    def __init__(self, wt_sequence: str, config: VAEConfig):
        super().__init__(); self.wt_sequence = wt_sequence; self.config = config; self.length = config.length; self.latent_dim = config.latent_dim
        self.aa_emb = nn.Embedding(len(AMINO_ACIDS)+1, config.hidden_dim); self.wt_emb = nn.Embedding(len(AMINO_ACIDS), config.hidden_dim); self.pos_emb = nn.Embedding(config.length, config.hidden_dim); self.mut_emb = nn.Embedding(2, config.hidden_dim)
        enc = nn.TransformerEncoderLayer(d_model=config.hidden_dim, nhead=4, dim_feedforward=config.hidden_dim*4, dropout=config.dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=config.num_layers); self.pool = nn.Linear(config.hidden_dim, 1); self.mu = nn.Linear(config.hidden_dim, config.latent_dim); self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.z_to_h = nn.Linear(config.latent_dim, config.hidden_dim)
        dec = nn.TransformerEncoderLayer(d_model=config.hidden_dim, nhead=4, dim_feedforward=config.hidden_dim*4, dropout=config.dropout, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(dec, num_layers=config.num_layers); self.aa_out = nn.Linear(config.hidden_dim, len(AMINO_ACIDS)); self.gate_out = nn.Linear(config.hidden_dim, 1)
        self.register_buffer('wt_tokens', torch.tensor([AA_TO_IDX[a] for a in wt_sequence], dtype=torch.long)); self.register_buffer('pos_ids', torch.arange(config.length, dtype=torch.long))
    def corrupt(self, x, mask_probability):
        if mask_probability <= 0: return x
        x_in = x.clone(); mask = torch.rand_like(x.float()) < mask_probability; x_in[mask] = MASK_IDX; return x_in
    def encode(self, x, mask_probability: float = 0.0):
        x_in = self.corrupt(x, mask_probability); bsz, length = x.shape; pos = self.pos_ids[None,:].expand(bsz, length); wt = self.wt_tokens[None,:].expand(bsz, length); mut = (x != wt).long()
        h = self.aa_emb(x_in)+self.wt_emb(wt)+self.pos_emb(pos)+self.mut_emb(mut); h = self.encoder(h); weights = torch.softmax(self.pool(h).squeeze(-1), dim=1); pooled = (h*weights.unsqueeze(-1)).sum(dim=1)
        return self.mu(pooled), self.logvar(pooled).clamp(-8,6)
    def reparameterize(self, mu, logvar):
        return mu if not self.training else mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
    def decode_logits(self, z):
        bsz = z.shape[0]; pos = self.pos_ids[None,:].expand(bsz, self.length); wt = self.wt_tokens[None,:].expand(bsz, self.length)
        h = self.z_to_h(z).unsqueeze(1).expand(bsz, self.length, -1); h = h+self.wt_emb(wt)+self.pos_emb(pos); h = self.decoder(h)
        return self.aa_out(h), self.gate_out(h).squeeze(-1)
    def forward(self, x, mask_probability: float = 0.15):
        mu, logvar = self.encode(x, mask_probability); z = self.reparameterize(mu, logvar); aa, gate = self.decode_logits(z); return aa, gate, mu, logvar
def vae_loss(model, x, mask_probability: float = 0.15, sample_weight=None):
    aa, gate, mu, logvar = model(x, mask_probability); ce = F.cross_entropy(aa.reshape(-1, len(AMINO_ACIDS)), x.reshape(-1), reduction='none').reshape(x.shape).mean(dim=1)
    if sample_weight is None: sample_weight = torch.ones_like(ce)
    ce_loss = (ce*sample_weight).sum()/sample_weight.sum().clamp(min=1.0); wt = model.wt_tokens[None,:].expand_as(x); target = (x != wt).float(); gate_loss = F.binary_cross_entropy_with_logits(gate, target, reduction='none').mean(dim=1); gate_loss = (gate_loss*sample_weight).sum()/sample_weight.sum().clamp(min=1.0)
    kl = -0.5*(1+logvar-mu.pow(2)-logvar.exp()).sum(dim=1).mean()/model.latent_dim; loss = ce_loss + model.config.mutation_gate_weight*gate_loss + model.config.kl_beta*kl
    return loss, {'ce': float(ce_loss.detach().cpu()), 'gate': float(gate_loss.detach().cpu()), 'kl': float(kl.detach().cpu())}, mu
