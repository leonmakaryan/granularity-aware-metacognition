# 7b: ESMA-style d'_type2 (direct answer = no-IDK prompt)


## locations test (303 questions)


### behavioural

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | 0.778 |  | 0.841 | 0.588 | 0.746 | 0.624 | direct IDK 0.000 |
| locations | 1.045 | +0.267 [-0.055, +0.599] | 0.743 | 0.347 | 0.607 | 0.658 | direct IDK 0.000 |
| dates | 0.789 | +0.011 [-0.225, +0.267] | 0.884 | 0.657 | 0.803 | 0.644 | direct IDK 0.000 |
| joint | 0.982 | +0.204 [-0.101, +0.517] | 0.751 | 0.381 | 0.625 | 0.659 | direct IDK 0.000 |

### explicit

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | 2.273 |  | 0.074 | 0.000 | 0.046 | 0.624 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC, AUROC 0.671, Yes/No mass 1.00 |
| locations | 2.149 | -0.124 [-0.249, -0.036] | 0.059 | 0.000 | 0.039 | 0.658 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC, AUROC 0.665, Yes/No mass 1.00 |
| dates | 1.800 | -0.473 [-0.649, +0.032] | 0.072 | 0.003 | 0.047 | 0.644 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC, AUROC 0.663, Yes/No mass 1.00 |
| joint | 1.988 | -0.285 [-0.543, -0.110] | 0.042 | 0.000 | 0.028 | 0.659 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC, AUROC 0.661, Yes/No mass 1.00 |

## dates test (503 questions)


### behavioural

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | 0.456 |  | 0.898 | 0.792 | 0.871 | 0.742 | direct IDK 0.000 |
| locations | 0.319 | -0.138 [-0.363, +0.083] | 0.817 | 0.721 | 0.797 | 0.796 | direct IDK 0.000 |
| dates | 1.078 | +0.621 [+0.272, +0.971] | 0.948 | 0.707 | 0.920 | 0.887 | direct IDK 0.000 |
| joint | 0.994 | +0.538 [+0.141, +0.923] | 0.937 | 0.704 | 0.912 | 0.893 | direct IDK 0.000 |
| dates_binary | 0.724 | +0.267 [+0.013, +0.525] | 0.879 | 0.672 | 0.844 | 0.830 | direct IDK 0.000 |

### explicit

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | -0.014 |  | 0.204 | 0.208 | 0.205 | 0.742 | direct IDK 0.000, AUROC 0.573, Yes/No mass 1.00 |
| locations | -0.028 | -0.015 [-0.244, +0.234] | 0.165 | 0.172 | 0.166 | 0.796 | direct IDK 0.000, AUROC 0.603, Yes/No mass 1.00 |
| dates | 0.742 | +0.756 [+0.350, +2.141] | 0.170 | 0.047 | 0.156 | 0.887 | direct IDK 0.000, AUROC 0.759, Yes/No mass 1.00 |
| joint | 1.414 | +1.428 [+1.032, +2.739] | 0.133 | 0.019 | 0.121 | 0.893 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC, AUROC 0.748, Yes/No mass 1.00 |
| dates_binary | 0.045 | +0.059 [-0.209, +0.359] | 0.188 | 0.176 | 0.186 | 0.830 | direct IDK 0.000, AUROC 0.651, Yes/No mass 1.00 |
