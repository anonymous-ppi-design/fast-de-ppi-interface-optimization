from dataclasses import dataclass
from typing import Sequence
@dataclass
class BudgetConfig:
    n_min:int=320; n_max:int=650; delta_max:float=0.62; fixed_round0:int=2209; fixed_round1:int=320
def cumulative_best(max_scores:Sequence[float])->list[float]:
    best=[]; cur=None
    for s in max_scores:
        cur=float(s) if cur is None else max(cur,float(s)); best.append(cur)
    return best
def next_budget(round_index:int, max_scores:Sequence[float], cfg:BudgetConfig=BudgetConfig())->int:
    if round_index==0: return cfg.fixed_round0
    if round_index==1: return cfg.fixed_round1
    if len(max_scores)<2: return cfg.n_max
    best=cumulative_best(max_scores); improvement=max(0.0,best[-1]-best[-2]); clipped=min(1.0, improvement/max(cfg.delta_max,1e-8)); raw=cfg.n_min+(cfg.n_max-cfg.n_min)*(1.0-clipped); return int(max(cfg.n_min,min(cfg.n_max,round(raw))))
def budget_schedule(max_scores:Sequence[float], cfg:BudgetConfig=BudgetConfig())->list[int]: return [next_budget(t,max_scores[:t],cfg) for t in range(len(max_scores))]
