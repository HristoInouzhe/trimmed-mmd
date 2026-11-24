# acs: ACS PUMS 2018 1-year, folktables ACSIncome, 51 states
groups: men n=62543, women n=57457; d=15; 512 random frequencies; bandwidth scale 1.0

## Lower bounds on TV of the unrepaired groups

| estimator | certified lower bound on TV |
|---|---|
| analytic test, alpha-hat = 0.250 | > 0.20 |
| sequential betting certificate (min over splits) | > 0.30 |
| cross-fitted GBM classifier (point estimate) | >= 0.380 |

Outcome (income > 50k): P(y|men)=0.439, P(y|women)=0.289; outcome classifier demographic-parity gap 0.073; largest |SMD| 0.51.

## Repaired cohorts at matched trimming level

audit TV = TPR-FPR of a cross-fitted GBM trained to recover the group (lower bound on the repaired groups' TV); split = split-sample certificate, min over splits, `none` = not even TV > 0 certified.

| alpha | method | ESS g0/g1 | max abs SMD | audit TV | split | outcome gap | clf DP gap |
|---|---|---|---|---|---|---|---|
| 0.150 | kernel trimming | 0.85/0.85 | 0.24 | 0.313 | – | +0.134 | 0.043 |
| 0.150 | propensity (GBM, cross-fit) | 0.85/0.85 | 0.40 | 0.283 | – | +0.140 | 0.053 |
| 0.150 | propensity (logit) | 0.85/0.85 | 0.42 | 0.326 | – | +0.138 | 0.053 |
| 0.150 | random deletion | 0.85/0.85 | 0.51 | 0.379 | – | +0.151 | 0.075 |
| 0.250 | kernel trimming | 0.75/0.75 | 0.06 | 0.277 | > 0.10 | +0.129 | 0.035 |
| 0.250 | propensity (GBM, cross-fit) | 0.75/0.75 | 0.20 | 0.235 | > 0.10 | +0.130 | 0.023 |
| 0.250 | propensity (logit) | 0.75/0.75 | 0.15 | 0.318 | – | +0.137 | 0.053 |
| 0.250 | random deletion | 0.75/0.75 | 0.52 | 0.382 | – | +0.149 | 0.071 |
| 0.350 | kernel trimming | 0.70/0.75 | 0.00 | 0.139 | none | +0.122 | 0.017 |
| 0.350 | propensity (GBM, cross-fit) | 0.65/0.65 | 0.17 | 0.193 | > 0.10 | +0.141 | 0.039 |
| 0.350 | propensity (logit) | 0.65/0.65 | 0.18 | 0.349 | – | +0.138 | 0.068 |
| 0.350 | random deletion | 0.65/0.65 | 0.51 | 0.374 | – | +0.150 | 0.073 |
| 0.400 | kernel trimming | 0.71/0.75 | 0.00 | 0.129 | none | +0.122 | 0.017 |
| 0.400 | propensity (GBM, cross-fit) | 0.60/0.60 | 0.16 | 0.196 | > 0.10 | +0.130 | 0.026 |
| 0.400 | propensity (logit) | 0.60/0.60 | 0.27 | 0.380 | – | +0.129 | 0.050 |
| 0.400 | random deletion | 0.60/0.60 | 0.52 | 0.378 | – | +0.149 | 0.075 |
| 0.500 | kernel trimming | 0.71/0.75 | 0.00 | 0.129 | none | +0.121 | 0.017 |
| 0.500 | propensity (GBM, cross-fit) | 0.50/0.50 | 0.30 | 0.295 | > 0.20 | +0.082 | 0.024 |
| 0.500 | propensity (logit) | 0.50/0.50 | 0.46 | 0.491 | – | +0.078 | 0.004 |
| 0.500 | random deletion | 0.50/0.50 | 0.51 | 0.372 | – | +0.150 | 0.077 |
