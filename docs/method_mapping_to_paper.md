# Method mapping to paper

## Section II-A: Target construct and structural prior

- `data/demo/demo_pretrain_sequences.csv`
- `configs/stage1_vae_pretrain.yaml`

ProteinMPNN itself is external. The repository demonstrates the processed table interface.

## Section II-B: ESM-regularized sparse-mutation VAE

- `src/fastde/models/sparse_mutation_vae.py`
- `src/fastde/training/train_stage1_vae.py`

The demo implementation is lightweight and runs without downloading ESM weights. Full-scale ESM embeddings can be supplied as external features.

## Section II-C: Physics-informed pseudo-labeling

- `src/fastde/scoring/foldx_parser.py`
- `src/fastde/scoring/foldx_composite_score.py`
- `configs/foldx_pseudolabel_score.yaml`

Implemented score:

```text
S(x) = -ddG_binding - 0.5 max(ddG_stability, 0) - 0.3 max(clash_increase, 0) - 0.05 mutation_count
```

## Section II-D: Random Frozen Head active optimization

- `src/fastde/models/frozen_heads.py`
- `src/fastde/models/trainable_head.py`
- `src/fastde/generation/latent_search.py`
- `src/fastde/training/train_stage2_active.py`

The demo compares no-head latent search, trainable head, and FAST-DE-RFH.

## Section II-E: Adaptive screening budget

- `src/fastde/scoring/budget_controller.py`
- `configs/adaptive_budget.yaml`
