# 7b: dev report


## locations test (303 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.752 | | +0.139 | | 0.521 | 0.254 | 0.224 | |
| in-domain | 0.778 | +0.025 [-0.023, +0.076] (includes 0) | +0.156 | +0.017 [-0.012, +0.046] (includes 0) | 0.486 | 0.393 | 0.121 | 0.012 / 0.002 |
| transfer | 0.708 | -0.044 [-0.077, -0.011] | +0.149 | +0.009 [-0.010, +0.030] (includes 0) | 0.573 | 0.197 | 0.230 | 0.008 / 0.001 |
| joint | 0.793 | +0.041 [-0.004, +0.087] (includes 0) | +0.168 | +0.029 [+0.001, +0.058] | 0.499 | 0.375 | 0.125 | 0.002 / 0.002 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.752 | 0.007 | 0.003 | 0.007 | 0.013 | 0.000 | 0.218 |
| in-domain | 0.778 | 0.019 | 0.002 | 0.003 | 0.066 | 0.014 | 0.118 |
| transfer | 0.708 | 0.023 | 0.000 | 0.007 | 0.000 | 0.039 | 0.223 |
| joint | 0.793 | 0.018 | 0.001 | 0.001 | 0.052 | 0.011 | 0.124 |

## dates test (503 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.746 | | +0.234 | | 0.648 | 0.129 | 0.223 | |
| in-domain | 0.695 | -0.050 [-0.093, -0.007] | +0.414 | +0.180 [+0.127, +0.235] | 0.814 | 0.080 | 0.107 | 0.010 / 0.002 |
| transfer | 0.722 | -0.023 [-0.049, +0.003] (includes 0) | +0.254 | +0.020 [-0.017, +0.056] (includes 0) | 0.622 | 0.203 | 0.176 | 0.008 / 0.011 |
| joint | 0.713 | -0.032 [-0.074, +0.009] (includes 0) | +0.430 | +0.196 [+0.143, +0.251] | 0.820 | 0.088 | 0.092 | 0.005 / 0.003 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.746 | 0.008 | 0.012 | 0.189 | 0.006 | 0.006 | 0.034 |
| in-domain | 0.695 | 0.120 | 0.023 | 0.078 | 0.000 | 0.055 | 0.028 |
| transfer | 0.722 | 0.021 | 0.008 | 0.144 | 0.072 | 0.001 | 0.032 |
| joint | 0.713 | 0.111 | 0.030 | 0.068 | 0.001 | 0.052 | 0.024 |
