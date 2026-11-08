"""Colab plumbing: Drive, .env, dataset download and SQLite backups."""

import os
import sys
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

DRIVE_ROOT = Path("/content/drive")
DRIVE_DIR = DRIVE_ROOT / "MyDrive" / "dl_lab2"


def mount_drive() -> bool:
    """Mount Google Drive and check that DRIVE_DIR is writable; False means results stay on the VM."""
    try:
        from google.colab import drive

        drive.mount(str(DRIVE_ROOT))
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        probe = DRIVE_DIR / ".write_test"
        probe.write_text("ok")
        assert probe.read_text() == "ok"
        probe.unlink()
        return True
    except Exception as error:
        print(f"Drive not available: {type(error).__name__}: {error}")
        return False


def load_env(path: str | Path = "/content/.env") -> list[str]:
    """Export KEY=VALUE lines to os.environ; returns the key names (values are never printed)."""
    keys = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'\"")
        keys.append(key.strip())
    return keys


def ensure_dataset(drive_dir: Path | None = None) -> None:
    """Make data/train, data/test and the csv files available, downloading from Kaggle if needed.

    With drive_dir the downloaded zip is kept there, so later sessions skip the Kaggle download.
    """
    from src.config import DATA_DIR
    if (DATA_DIR / "train").is_dir() and (DATA_DIR / "test.csv").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "cats-and-dogs.zip"
    saved = drive_dir / "cats-and-dogs.zip" if drive_dir else None
    if saved is not None and saved.exists():
        shutil.copy(saved, archive)
    else:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "cats-and-dogs", "-p", str(DATA_DIR)],
            check=True,
        )
        if saved is not None:
            shutil.copy(archive, saved)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(DATA_DIR)], check=True)
    archive.unlink()
    nested = DATA_DIR / "cats-and-dogs"  # some archives wrap everything in one folder
    if nested.is_dir() and not (DATA_DIR / "train").exists():
        for child in nested.iterdir():
            shutil.move(child, DATA_DIR / child.name)


def backup_sqlite(source: Path, destination: Path) -> None:
    """Consistent copy of a live SQLite file (plain file copies can catch it mid-write)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(tmp)) as dst:
        src.backup(dst)
    os.replace(tmp, destination)


def restore_sqlite(destination: Path, backup: Path) -> bool:
    """Bring a backed-up database back after a Colab restart, unless a local one exists."""
    if destination.exists() or not backup.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(backup, destination)
    return True


def in_colab() -> bool:
    return "google.colab" in sys.modules


def setup() -> Path | None:
    if not in_colab():
        return None
    
    drive_ok = mount_drive()
    os.environ.setdefault("DATA_DIR", "/content/data")
    os.environ.setdefault("MODELS_DIR", "/content/models")
    results = DRIVE_DIR / "results" if drive_ok else Path("/content/results")
    os.environ.setdefault("RESULTS_DIR", str(results))

    if "src.config" in sys.modules:
        raise RuntimeError("src.config was imported before colab.setup(): restart the session and run setup() first")

    env_file = DRIVE_DIR / ".env"
    if drive_ok and env_file.exists():
        print("secrets from Drive:", load_env(env_file))
    ensure_dataset(DRIVE_DIR if drive_ok else None)

    from src.config import DATA_DIR, MODELS_DIR, RESULTS_DIR
    print(f"DATA_DIR={DATA_DIR}\nMODELS_DIR={MODELS_DIR}\nRESULTS_DIR={RESULTS_DIR}")
    return DRIVE_DIR if drive_ok else None    