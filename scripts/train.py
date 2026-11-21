"""Train a NoPeWorld head on a frozen world model. Same as the console script `nopeworld-train`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nopeworld.cli import train_main  # noqa: E402

if __name__ == "__main__":
    train_main()
