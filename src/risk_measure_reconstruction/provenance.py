"""Run identity, atomic output, and a single-writer guard."""

from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys


def runtime_versions():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "pandas", "clarabel", "mpmath", "matplotlib")
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def check_config(path, config):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(
            f"Configuration, source, or runtime differs: use a new output directory ({path})."
        )
    write_json(path, config)


@contextmanager
def run_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"Output directory is locked: {lock}. If a previous process crashed, "
            "verify it is no longer running before removing this lock."
        ) from exc
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"pid": os.getpid(), "command": sys.argv}, handle)
        yield out
    finally:
        lock.unlink(missing_ok=True)
