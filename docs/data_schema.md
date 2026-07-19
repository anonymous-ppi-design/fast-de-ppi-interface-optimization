# Data schema

## Pretraining sequence table

Required columns:

```text
design_sequence
mutation_count
pretrain_source
```

## FoldX pseudo-label table

Required columns:

```text
candidate_id
design_sequence
mutation_count
ddG_binding_vs_WT
ddG_stability_vs_WT
clash_increase_vs_WT
is_WT
```

## Active-round metrics

Produced by Stage 2:

```text
method
round
n
mean_score
max_score
top20_mean_score
median_mutation_count
```
