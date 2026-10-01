"""The final pipeline: sample -> target -> train -> evaluate.

Every stage uses the same model settings, prompts, generation code and scorer.

    config.py        every setting
    data.py          one question type for every evaluation set
    score.py         the scorer (evidence in samples, training-answer checks, grading)
    generate.py      model loading and generation
    targets.py       the 50%-support rule and the training records
    train.py         LoRA fine-tuning
    support.py       support-match categories
    report.py        tables, paired bootstrap, manifest
    meta.py, esma.py the ESMA-style self-knowledge test
    probes.py        conflicting sources and unseen entity relations
    scorer_check.py  Gate 2: the scorer against hand labels
    checks.py        Gate 1: offline tests of the scoring rules
    run.py           command line for every stage
"""

import sys
from pathlib import Path

# Lukas Ellinger's temporal parser (ugc) and level widths (evaluation) live in third_party/.
_THIRD_PARTY = str(Path(__file__).resolve().parent.parent / "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
