# 1p5b: dev report


## locations test (303 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.729 | | +0.095 | | 0.416 | 0.389 | 0.195 | |
| in-domain | 0.750 | +0.021 [-0.030, +0.070] (includes 0) | +0.120 | +0.025 [-0.011, +0.062] (includes 0) | 0.389 | 0.508 | 0.102 | 0.006 / 0.005 |
| transfer | 0.713 | -0.017 [-0.057, +0.023] (includes 0) | +0.115 | +0.020 [-0.009, +0.049] (includes 0) | 0.480 | 0.334 | 0.186 | 0.012 / 0.002 |
| joint | 0.733 | +0.003 [-0.045, +0.051] (includes 0) | +0.112 | +0.018 [-0.019, +0.055] (includes 0) | 0.361 | 0.540 | 0.099 | 0.005 / 0.007 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.729 | 0.007 | 0.007 | 0.023 | 0.046 | 0.017 | 0.172 |
| in-domain | 0.750 | 0.018 | 0.000 | 0.011 | 0.099 | 0.031 | 0.091 |
| transfer | 0.713 | 0.025 | 0.000 | 0.012 | 0.026 | 0.050 | 0.174 |
| joint | 0.733 | 0.017 | 0.000 | 0.011 | 0.124 | 0.028 | 0.088 |

## dates test (503 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.539 | | -0.139 | | 0.227 | 0.431 | 0.342 | |
| in-domain | 0.496 | -0.043 [-0.095, +0.009] (includes 0) | +0.194 | +0.333 [+0.271, +0.395] | 0.563 | 0.254 | 0.182 | 0.002 / 0.005 |
| transfer | 0.522 | -0.017 [-0.050, +0.017] (includes 0) | -0.105 | +0.034 [-0.010, +0.078] (includes 0) | 0.126 | 0.605 | 0.269 | 0.003 / 0.009 |
| joint | 0.513 | -0.026 [-0.077, +0.025] (includes 0) | +0.188 | +0.327 [+0.266, +0.389] | 0.545 | 0.277 | 0.178 | 0.003 / 0.007 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.539 | 0.004 | 0.016 | 0.245 | 0.076 | 0.024 | 0.097 |
| in-domain | 0.496 | 0.093 | 0.040 | 0.101 | 0.024 | 0.165 | 0.082 |
| transfer | 0.522 | 0.006 | 0.003 | 0.219 | 0.189 | 0.011 | 0.050 |
| joint | 0.513 | 0.094 | 0.036 | 0.101 | 0.028 | 0.152 | 0.076 |
