# Adversarial and boundary null pairs (revision v2)

- `validity_reps.jsonl`: verbatim copy of `explore/sharper_threshold/out/validity_reps.jsonl` (sha256 `c0f6ffca…69e0`),
  7,000 replicates of D-hat at n = m = 300 on 19 null pairs with TV(P,Q) = alpha* exactly (seeds 10000+i; 13 designs
  x 400, then the six adversarial atom designs x 300). Producer: `scripts/adversarial_nulls.py run` (the research
  code, copied unchanged in computation; three replicates re-run bit-identical on 2026-11-08). Threshold-independent.
- Read by `jmlr/tools/make_constants.py` through `scripts/adversarial_nulls.py:floor_table` for `\floorRatioLo`,
  `\floorRatioHi`: per alpha, tau_new(300, 300, alpha, 0.05) (Gaussian constants) over the largest 95% quantile of
  D-hat among the atom designs at that alpha. Values: 1.89 (alpha = 0), 3.38 (0.05), 3.51 (0.2), 3.74 (0.5).
  `PYTHONPATH=src python3 scripts/adversarial_nulls.py summarize` prints the table.
