import mlflow

from src.config import ROOT

MLRUNS_DIR = ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"
ARTIFACTS_DIR = MLRUNS_DIR / "artifacts"
EXPERIMENT_NAME = "dl_lab-2"


def setup():
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
