# FoldX setup note

This repository does not redistribute FoldX.

For full runs, install FoldX separately under its own license. FAST-DE expects a merged FoldX pseudo-label table with columns:

```text
candidate_id
design_sequence
mutation_count
ddG_binding_vs_WT
ddG_stability_vs_WT
clash_increase_vs_WT
is_WT
```

`src/fastde/scoring/foldx_composite_score.py` converts those terms into the active-learning pseudo-label used by Stage 2.
