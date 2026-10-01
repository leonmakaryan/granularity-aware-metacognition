# 3b: dev report


## locations test (303 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.568 | | +0.095 | | 0.564 | 0.050 | 0.386 | |
| in-domain | 0.689 | +0.121 [+0.059, +0.184] | +0.184 | +0.089 [+0.041, +0.136] | 0.534 | 0.309 | 0.157 | 0.010 / 0.005 |
| transfer | 0.582 | +0.014 [-0.009, +0.037] (includes 0) | +0.105 | +0.010 [-0.008, +0.030] (includes 0) | 0.553 | 0.088 | 0.359 | 0.004 / 0.002 |
| joint | 0.677 | +0.109 [+0.046, +0.172] | +0.173 | +0.078 [+0.028, +0.127] | 0.510 | 0.330 | 0.160 | 0.003 / 0.002 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.568 | 0.010 | 0.013 | 0.046 | 0.000 | 0.023 | 0.340 |
| in-domain | 0.689 | 0.051 | 0.008 | 0.011 | 0.069 | 0.026 | 0.146 |
| transfer | 0.582 | 0.015 | 0.010 | 0.042 | 0.013 | 0.021 | 0.317 |
| joint | 0.677 | 0.047 | 0.008 | 0.017 | 0.085 | 0.024 | 0.143 |

## dates test (503 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.720 | | -0.003 | | 0.225 | 0.640 | 0.135 | |
| in-domain | 0.630 | -0.090 [-0.136, -0.045] | +0.058 | +0.062 [+0.045, +0.078] | 0.445 | 0.434 | 0.121 | 0.022 / 0.003 |
| transfer | 0.724 | +0.004 [-0.044, +0.051] (includes 0) | -0.004 | -0.001 [-0.016, +0.015] (includes 0) | 0.027 | 0.953 | 0.021 | 0.002 / 0.001 |
| joint | 0.643 | -0.077 [-0.124, -0.032] | +0.059 | +0.062 [+0.047, +0.078] | 0.443 | 0.443 | 0.114 | 0.034 / 0.005 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.720 | 0.000 | 0.006 | 0.014 | 0.085 | 0.054 | 0.121 |
| in-domain | 0.630 | 0.006 | 0.004 | 0.013 | 0.026 | 0.213 | 0.108 |
| transfer | 0.724 | 0.000 | 0.001 | 0.007 | 0.246 | 0.009 | 0.013 |
| joint | 0.643 | 0.005 | 0.001 | 0.013 | 0.026 | 0.211 | 0.101 |
