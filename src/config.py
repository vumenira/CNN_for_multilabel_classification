from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FOLDS_PATH = DATA_DIR / "folds.csv"
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]
SEED = 42
