import sys
from pathlib import Path

# Put "<repo>/src" at the front of sys.path so "import scripts" works in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
