import sys
from pathlib import Path

# Lukas Ellinger's temporal parser (ugc) and level widths (evaluation) live in third_party/.
_THIRD_PARTY = str(Path(__file__).resolve().parents[2] / "third_party")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
