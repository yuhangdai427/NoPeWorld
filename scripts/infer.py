"""Generate with a world model and a method, or score Task 1. Same as the console script `nopeworld-infer`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nopeworld.cli import infer_main  # noqa: E402

if __name__ == "__main__":
    infer_main()
