# 1p5b: dev report


## locations test (303 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.686 | | +0.070 | | 0.495 | 0.231 | 0.274 | |
| in-domain | 0.653 | -0.033 [-0.083, +0.017] (includes 0) | +0.110 | +0.040 [-0.001, +0.082] (includes 0) | 0.531 | 0.222 | 0.246 | 0.000 / 0.000 |
| transfer | 0.605 | -0.081 [-0.125, -0.039] | +0.089 | +0.019 [-0.013, +0.054] (includes 0) | 0.560 | 0.129 | 0.311 | 0.012 / 0.007 |
| joint | 0.660 | -0.026 [-0.076, +0.023] (includes 0) | +0.107 | +0.037 [-0.005, +0.078] (includes 0) | 0.540 | 0.213 | 0.246 | 0.007 / 0.004 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.686 | 0.007 | 0.003 | 0.043 | 0.010 | 0.020 | 0.231 |
| in-domain | 0.653 | 0.041 | 0.003 | 0.022 | 0.015 | 0.041 | 0.224 |
| transfer | 0.605 | 0.030 | 0.000 | 0.022 | 0.000 | 0.054 | 0.289 |
| joint | 0.660 | 0.039 | 0.003 | 0.024 | 0.008 | 0.044 | 0.222 |

## dates test (503 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.372 | | -0.225 | | 0.334 | 0.117 | 0.549 | |
| in-domain | 0.367 | -0.005 [-0.061, +0.053] (includes 0) | +0.204 | +0.429 [+0.363, +0.497] | 0.694 | 0.001 | 0.305 | 0.008 / 0.003 |
| transfer | 0.345 | -0.027 [-0.063, +0.009] (includes 0) | -0.171 | +0.054 [+0.002, +0.105] | 0.248 | 0.270 | 0.482 | 0.007 / 0.002 |
| joint | 0.379 | +0.007 [-0.049, +0.065] (includes 0) | +0.218 | +0.443 [+0.376, +0.510] | 0.707 | 0.004 | 0.289 | 0.005 / 0.001 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.372 | 0.004 | 0.024 | 0.408 | 0.032 | 0.020 | 0.141 |
| in-domain | 0.367 | 0.158 | 0.049 | 0.180 | 0.000 | 0.121 | 0.125 |
| transfer | 0.345 | 0.004 | 0.018 | 0.376 | 0.140 | 0.011 | 0.107 |
| joint | 0.379 | 0.158 | 0.052 | 0.169 | 0.000 | 0.123 | 0.120 |
