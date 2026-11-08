import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default

DATA_DIR    = _path_from_env("DATA_DIR", ROOT / "data")
FOLDS_PATH  = ROOT/ "data" / "folds.csv"
MODELS_DIR  = _path_from_env("MODELS_DIR", ROOT / "models")
RESULTS_DIR = _path_from_env("RESULTS_DIR", ROOT / "results")

LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
SEED = 42
