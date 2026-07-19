# FAST-DE: Feedback-Adaptive Sparse Targeted Directed Evolution

This is an anonymized review repository for **FAST-DE**, a target-specific active latent directed-evolution framework for protein-protein interface optimization.

The repository is intentionally lightweight. It provides code, configuration files, small demo tables, and scripts that show how the method described in the paper can be run end-to-end. It does **not** redistribute FoldX, ProteinMPNN, RFdiffusion, ESM model weights, large checkpoints, or full paper-result tables.

## Included components

1. sparse-mutation VAE training,
2. FoldX-derived pseudo-label table validation,
3. FoldX composite active-learning score,
4. active latent optimization with no-head, trainable-head, and Random Frozen Head (RFH) strategies,
5. adaptive screening-budget control,
6. active-round benchmark summaries.

The repository follows the paper pipeline:

```text
target-specific sequence prior
→ sparse-mutation VAE
→ FoldX / physics-informed pseudo-labels
→ active latent optimization
→ no-head vs trainable-head vs FAST-DE-RFH
→ adaptive screening budget
```

## Anonymous review note

This repository is anonymized for double-blind review. It contains no author names, institutional paths, private cloud links, FoldX executable, or external model checkpoints.

## Quick demo

The demo uses small processed example tables with the same schema expected by the full pipeline. It verifies that the code path runs; it is not intended to reproduce the full paper results.

```bash
conda env create -f environment.yml
conda activate fastde
bash scripts/00_run_demo.sh
```

Expected outputs are written to `outputs/demo/`.

## Full pipeline commands

```bash
python -m fastde.training.train_stage1_vae \
  --config configs/stage1_vae_pretrain.yaml \
  --input data/demo/demo_pretrain_sequences.csv \
  --out outputs/demo/stage1

python -m fastde.scoring.foldx_composite_score \
  --input data/demo/demo_foldx_pseudolabels.csv \
  --config configs/foldx_pseudolabel_score.yaml \
  --out outputs/demo/demo_foldx_scored.csv

python -m fastde.training.train_stage2_active \
  --config configs/stage2_active_training.yaml \
  --vae-checkpoint outputs/demo/stage1/stage1_vae.pt \
  --labels outputs/demo/demo_foldx_scored.csv \
  --out outputs/demo/stage2
```

## External dependencies

The full paper pipeline uses external tools that must be installed separately under their own licenses: FoldX, ProteinMPNN, RFdiffusion, and ESM. This repository provides FAST-DE code, wrappers, and schemas, but does not redistribute those tools or weights.

## License

See `LICENSE`.
