"""Download the heads and the dataset. Same as the console script `nopebench-download`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nopebench.cli import download_main  # noqa: E402

if __name__ == "__main__":
    download_main()
