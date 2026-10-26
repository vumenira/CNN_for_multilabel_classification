import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FOLDS_PATH = DATA_DIR / "folds.csv"
# Colab sets RESULTS_DIR to a Drive folder so that run exports survive the session.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
SEED = 42
