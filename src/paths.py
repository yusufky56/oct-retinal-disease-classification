"""Location of the Kermany OCT2017 dataset.

Set the OCT_DATA_ROOT environment variable to the folder that contains
`train/`, `test/` and `val/`. Defaults to `data/OCT2017` in the repo.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("OCT_DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "OCT2017"))


def resolve_image(filepath: str) -> Path:
    """Split CSVs store paths relative to DATA_ROOT; absolute paths are used as is."""
    path = Path(filepath)
    return path if path.is_absolute() else DATA_ROOT / path
