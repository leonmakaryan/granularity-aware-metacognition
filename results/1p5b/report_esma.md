# 1p5b: ESMA-style d'_type2 (direct answer = no-IDK prompt)


## locations test (303 questions)


### behavioural

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | 1.091 |  | 0.923 | 0.631 | 0.769 | 0.472 | direct IDK 0.003 |
| locations | 1.168 | +0.077 [-0.389, +0.512] | 0.919 | 0.591 | 0.778 | 0.570 | direct IDK 0.000 |
| dates | 1.179 | +0.088 [-0.386, +0.831] | 0.968 | 0.755 | 0.871 | 0.546 | direct IDK 0.000 |
| joint | 1.170 | +0.079 [-0.402, +0.537] | 0.924 | 0.606 | 0.787 | 0.568 | direct IDK 0.000 |

### explicit

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | -0.117 |  | 0.014 | 0.019 | 0.017 | 0.472 | direct IDK 0.003, AUROC 0.630, Yes/No mass 0.99 |
| locations | 0.113 | +0.231 [-1.347, +2.099] | 0.023 | 0.018 | 0.021 | 0.570 | direct IDK 0.000, AUROC 0.630, Yes/No mass 0.99 |
| dates | -0.236 | -0.119 [-0.173, -0.043] | 0.012 | 0.022 | 0.017 | 0.546 | direct IDK 0.000, AUROC 0.608, Yes/No mass 0.99 |
| joint | 0.292 | +0.410 [-0.525, +2.411] | 0.023 | 0.013 | 0.019 | 0.568 | direct IDK 0.000, AUROC 0.625, Yes/No mass 1.00 |

## dates test (503 questions)


### behavioural

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | 0.164 |  | 0.902 | 0.871 | 0.883 | 0.386 | direct IDK 0.018 |
| locations | 0.143 | -0.021 [-0.317, +0.255] | 0.757 | 0.711 | 0.730 | 0.428 | direct IDK 0.046 |
| dates | 0.811 | +0.646 [-0.390, +1.062] | 1.000 | 0.996 | 0.999 | 0.677 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC |
| joint | 1.491 | +1.326 [-0.268, +1.774] | 1.000 | 0.987 | 0.996 | 0.693 | direct IDK 0.000, d' AT THE CLIP (a rate is 0 or 1): use AUROC |
| dates_binary | 0.649 | +0.484 [+0.107, +0.844] | 0.899 | 0.737 | 0.844 | 0.660 | direct IDK 0.000 |

### explicit

| condition | d' | delta vs base [95% CI] | hit rate | false alarm | yes rate | direct acc | extra |
|---|---|---|---|---|---|---|---|
| base | 0.587 |  | 0.320 | 0.146 | 0.213 | 0.386 | direct IDK 0.018, AUROC 0.679, Yes/No mass 0.99 |
| locations | 0.609 | +0.022 [-0.153, +0.195] | 0.412 | 0.203 | 0.292 | 0.428 | direct IDK 0.046, AUROC 0.676, Yes/No mass 1.00 |
| dates | 0.924 | +0.338 [+0.010, +0.735] | 0.285 | 0.068 | 0.215 | 0.677 | direct IDK 0.000, AUROC 0.722, Yes/No mass 0.99 |
| joint | 0.992 | +0.406 [+0.079, +0.785] | 0.355 | 0.086 | 0.272 | 0.693 | direct IDK 0.000, AUROC 0.722, Yes/No mass 0.99 |
| dates_binary | 1.354 | +0.767 [+0.428, +1.203] | 0.366 | 0.045 | 0.256 | 0.660 | direct IDK 0.000, AUROC 0.769, Yes/No mass 0.99 |
