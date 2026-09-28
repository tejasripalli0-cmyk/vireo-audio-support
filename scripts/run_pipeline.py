"""Thin wrapper so `python scripts/run_pipeline.py` also works.
The canonical entry point is run.py in the project root -- see README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import main

if __name__ == "__main__":
    main()
