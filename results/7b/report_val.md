# 7b: val report


## locations val (285 questions)

| Condition | Support match | Delta [95% CI] | Signed score | Delta [95% CI] | Correct | Abstain | Wrong | Seed SD (match / score) |
|---|---|---|---|---|---|---|---|---|
| base | 0.754 | | +0.084 | | 0.516 | 0.253 | 0.232 | |
| in-domain | 0.777 | +0.022 [-0.026, +0.070] (includes 0) | +0.116 | +0.032 [+0.005, +0.062] | 0.460 | 0.396 | 0.144 | 0.006 / 0.002 |
| transfer | 0.709 | -0.046 [-0.080, -0.014] | +0.085 | +0.001 [-0.021, +0.024] (includes 0) | 0.538 | 0.205 | 0.257 | 0.010 / 0.004 |
| joint | 0.765 | +0.011 [-0.037, +0.057] (includes 0) | +0.109 | +0.025 [-0.003, +0.054] (includes 0) | 0.467 | 0.384 | 0.150 | 0.005 / 0.001 |

Estimated-support breakdown (share of questions):

| Condition | exact_support_match | correct_coarser_than_support | correct_finer_than_support | wrong_despite_support | abstained_despite_support | answered_without_support_correct | answered_without_support_wrong |
|---|---|---|---|---|---|---|---|
| base | 0.754 | 0.004 | 0.000 | 0.018 | 0.004 | 0.007 | 0.214 |
| in-domain | 0.777 | 0.006 | 0.001 | 0.000 | 0.071 | 0.001 | 0.144 |
| transfer | 0.709 | 0.016 | 0.000 | 0.009 | 0.000 | 0.018 | 0.248 |
| joint | 0.765 | 0.019 | 0.001 | 0.000 | 0.064 | 0.001 | 0.150 |
