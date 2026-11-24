# adult: UCI Adult (Becker & Kohavi 1996), train + test
groups: men n=30527, women n=14695; d=12; 512 random frequencies; bandwidth scale 1.0

## Lower bounds on TV of the unrepaired groups

| estimator | certified lower bound on TV |
|---|---|
| analytic test, alpha-hat = 0.175 | > 0.15 |
| split-sample certificate (min over splits) | > 0.40 |
| cross-fitted GBM classifier (point estimate) | >= 0.496 |

Outcome (income > 50k): P(y|men)=0.312, P(y|women)=0.114; outcome classifier demographic-parity gap 0.184; largest |SMD| 0.91.

## Repaired cohorts at matched trimming level

audit TV = TPR-FPR of a cross-fitted GBM trained to recover the group (lower bound on the repaired groups' TV); split = split-sample certificate, min over splits, `none` = not even TV > 0 certified.

| alpha | method | ESS g0/g1 | max abs SMD | audit TV | split | outcome gap | clf DP gap |
|---|---|---|---|---|---|---|---|
| 0.050 | kernel trimming | 0.95/0.95 | 0.86 | 0.485 | – | +0.183 | 0.173 |
| 0.050 | propensity (GBM, cross-fit) | 0.95/0.95 | 0.86 | 0.472 | – | +0.182 | 0.167 |
| 0.050 | propensity (logit) | 0.95/0.95 | 0.86 | 0.473 | – | +0.187 | 0.180 |
| 0.050 | random deletion | 0.95/0.95 | 0.92 | 0.495 | – | +0.200 | 0.189 |
| 0.100 | kernel trimming | 0.90/0.90 | 0.79 | 0.474 | – | +0.162 | 0.154 |
| 0.100 | propensity (GBM, cross-fit) | 0.90/0.90 | 0.79 | 0.440 | – | +0.166 | 0.158 |
| 0.100 | propensity (logit) | 0.90/0.90 | 0.79 | 0.451 | – | +0.173 | 0.165 |
| 0.100 | random deletion | 0.90/0.90 | 0.91 | 0.498 | – | +0.199 | 0.186 |
| 0.175 | kernel trimming | 0.83/0.83 | 0.68 | 0.451 | > 0.30 | +0.131 | 0.125 |
| 0.175 | propensity (GBM, cross-fit) | 0.83/0.82 | 0.69 | 0.400 | > 0.30 | +0.139 | 0.132 |
| 0.175 | propensity (logit) | 0.83/0.82 | 0.68 | 0.428 | – | +0.150 | 0.148 |
| 0.175 | random deletion | 0.83/0.82 | 0.92 | 0.500 | – | +0.199 | 0.187 |
| 0.250 | kernel trimming | 0.75/0.75 | 0.55 | 0.419 | – | +0.102 | 0.106 |
| 0.250 | propensity (GBM, cross-fit) | 0.75/0.75 | 0.57 | 0.356 | – | +0.113 | 0.102 |
| 0.250 | propensity (logit) | 0.75/0.75 | 0.55 | 0.402 | – | +0.126 | 0.122 |
| 0.250 | random deletion | 0.75/0.75 | 0.90 | 0.493 | – | +0.199 | 0.183 |
| 0.350 | kernel trimming | 0.65/0.65 | 0.33 | 0.382 | > 0.15 | +0.055 | 0.065 |
| 0.350 | propensity (GBM, cross-fit) | 0.65/0.65 | 0.36 | 0.291 | > 0.15 | +0.070 | 0.056 |
| 0.350 | propensity (logit) | 0.65/0.65 | 0.42 | 0.387 | – | +0.084 | 0.088 |
| 0.350 | random deletion | 0.65/0.65 | 0.90 | 0.489 | – | +0.198 | 0.182 |
| 0.450 | kernel trimming | 0.55/0.56 | 0.03 | 0.312 | none | -0.003 | 0.001 |
| 0.450 | propensity (GBM, cross-fit) | 0.55/0.55 | 0.18 | 0.221 | > 0.05 | +0.012 | 0.018 |
| 0.450 | propensity (logit) | 0.55/0.55 | 0.28 | 0.396 | – | +0.038 | 0.063 |
| 0.450 | random deletion | 0.55/0.55 | 0.92 | 0.500 | – | +0.203 | 0.183 |
| 0.500 | kernel trimming | 0.58/0.65 | 0.00 | 0.111 | none | +0.002 | 0.001 |
| 0.500 | propensity (GBM, cross-fit) | 0.50/0.50 | 0.10 | 0.233 | > 0.05 | -0.025 | 0.025 |
| 0.500 | propensity (logit) | 0.50/0.50 | 0.18 | 0.408 | – | +0.001 | 0.027 |
| 0.500 | random deletion | 0.50/0.50 | 0.91 | 0.500 | – | +0.200 | 0.185 |
| 0.550 | kernel trimming | 0.58/0.64 | 0.00 | 0.104 | – | +0.002 | 0.006 |
| 0.550 | propensity (GBM, cross-fit) | 0.45/0.45 | 0.30 | 0.263 | – | -0.065 | 0.070 |
| 0.550 | propensity (logit) | 0.45/0.45 | 0.39 | 0.447 | – | -0.055 | 0.044 |
| 0.550 | random deletion | 0.45/0.45 | 0.91 | 0.493 | – | +0.203 | 0.192 |
| 0.600 | kernel trimming | 0.58/0.64 | 0.00 | 0.106 | none | +0.002 | 0.002 |
| 0.600 | propensity (GBM, cross-fit) | 0.40/0.40 | 0.52 | 0.312 | > 0.20 | -0.101 | 0.104 |
| 0.600 | propensity (logit) | 0.40/0.40 | 0.67 | 0.489 | – | -0.111 | 0.112 |
| 0.600 | random deletion | 0.40/0.40 | 0.92 | 0.486 | – | +0.192 | 0.176 |
| 0.700 | kernel trimming | 0.58/0.64 | 0.00 | 0.110 | – | +0.002 | 0.000 |
| 0.700 | propensity (GBM, cross-fit) | 0.30/0.30 | 0.81 | 0.645 | – | -0.156 | 0.162 |
| 0.700 | propensity (logit) | 0.30/0.30 | 1.12 | 0.784 | – | -0.197 | 0.189 |
| 0.700 | random deletion | 0.30/0.30 | 0.90 | 0.479 | – | +0.190 | 0.187 |

## Classical propensity rules (they choose their own level)

| score | rule | cutoff | kept g0/g1 | max abs SMD | audit TV | outcome gap | clf DP gap |
|---|---|---|---|---|---|---|---|
| logit | fixed [0.1, 0.9] | 0.100 | 0.74/0.97 | 0.71 | 0.419 | +0.156 | 0.151 |
| logit | Crump cutoff | 0.089 | 0.79/0.98 | 0.76 | 0.440 | +0.165 | 0.159 |
| GBM | fixed [0.1, 0.9] | 0.100 | 0.61/0.95 | 0.52 | 0.364 | +0.103 | 0.083 |
| GBM | Crump cutoff | 0.101 | 0.61/0.95 | 0.52 | 0.361 | +0.102 | 0.084 |
