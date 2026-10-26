"""Colab plumbing: Drive, .env, dataset download and SQLite backups."""

import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

from src.config import DATA_DIR

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
    if (DATA_DIR / "train").is_dir() and (DATA_DIR / "test.csv").exists():
        return
    DATA_DIR.mkdir(exist_ok=True)
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
