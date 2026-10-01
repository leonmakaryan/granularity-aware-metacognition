"""Rewards for the GRPO runs.

The level reward is paid only when the answer is correct, so a coarse wrong answer
earns nothing for being coarse. From best to worst: correct at the target level,
correct at another level, "I don't know" when the model is uncertain, "I don't know"
when it should know, and a wrong answer.
"""

import math
from collections import Counter

from .config import (IDK_REWARD, IDK_WHEN_KNOW, WRONG_PENALTY, LEVEL_FLOOR,
                     LEVEL_DIFF_K, W_LEVEL, W_GRAN, MAX_ANSWER_WORDS, BREVITY_PENALTY)
from .text import find_match_level, normalize
from .prompt import is_idk


def _too_long(ans: str) -> bool:
    return len(ans.split()) > MAX_ANSWER_WORDS


def reward_b2(
    ans: str,
    hierarchy: list[str],
    levels: list[float],
    target_level: int,
    target_bin: int,
    high_entropy_bin: int,
    measured_gs: float | None = None,
    target_gs: float | None = None,
) -> float:
    """Single-answer reward. `measured_gs`/`target_gs` drive the small secondary
    GranuScore term; pass None (or set W_GRAN=0) to use level-closeness only.
    """
    idk = is_idk(ans)
    ml = find_match_level(ans, hierarchy) if not idk else None

    # Abstention is rewarded only at the highest-entropy bin (knows nothing).
    if idk:
        return IDK_REWARD if target_bin >= high_entropy_bin else IDK_WHEN_KNOW

    # Confident but off-hierarchy: this is the failure we are killing.
    if ml is None:
        return WRONG_PENALTY

    # On-hierarchy: pay for closeness to the entropy-derived target level,
    # measured in normalized-level space (Lukas's normalized_levels).
    n = len(levels)
    target_norm = levels[min(target_level, n - 1)] if n else 0.0
    match_norm = levels[ml] if ml < n else levels[-1]
    level_close = max(LEVEL_FLOOR, 1.0 - LEVEL_DIFF_K * abs(match_norm - target_norm))

    reward = W_LEVEL * level_close

    # Small secondary GranuScore term (also gated: only added when on-hierarchy).
    if W_GRAN and measured_gs is not None and target_gs is not None:
        gran_close = max(0.0, 1.0 - abs(measured_gs - target_gs) / 100.0) ** 2
        reward += W_GRAN * gran_close

    if _too_long(ans):
        reward -= BREVITY_PENALTY
    return reward


def reward_specificity(
    ans: str,
    hierarchy: list[str],
    levels: list[float],
    target_level: int,
    target_bin: int,
    high_entropy_bin: int,
    measured_gs: float | None = None,
    target_gs: float | None = None,
) -> float:
    """Variant without a target level: the finer the correct answer, the higher the
        reward (city 1.0, region 0.67, country 0.33). Same gate and abstention terms."""
    idk = is_idk(ans)
    ml = find_match_level(ans, hierarchy) if not idk else None
    if idk:
        return IDK_REWARD if target_bin >= high_entropy_bin else IDK_WHEN_KNOW
    if ml is None:
        return WRONG_PENALTY
    n = len(hierarchy)
    specificity = (n - 1 - ml) / (n - 1) if n > 1 else 1.0
    r = max(LEVEL_FLOOR, specificity)
    return r - (BREVITY_PENALTY if _too_long(ans) else 0.0)


def semantic_entropy(answers: list[str]) -> float:
    keys = [normalize(a) if a else "__empty__" for a in answers]
    counts = Counter(keys)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log(p) for p in probs if p > 0)


def entropy_to_bin(entropy: float, bins: list[float], n_levels: int) -> int:
    for i, b in enumerate(bins):
        if entropy <= b:
            return i
    return n_levels - 1
