"""Evaluate a run on Task 1, 2 or 3. Same as the console script `nopebench-eval`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nopebench.cli import eval_main  # noqa: E402

if __name__ == "__main__":
    eval_main()
