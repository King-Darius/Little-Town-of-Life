"""Convenience launcher for the Smallville+++ prototype.

The script performs a very small bootstrap step so that curious players can
double-click (or run ``python run_smallville.py``) and immediately explore the
simulation without manually installing Python packages.  It installs lightweight
dependencies on-demand before importing the GUI module.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "Pillow": "PIL",
    "numpy": "numpy",
    "huggingface_hub": "huggingface_hub",
    "llama-cpp-python": "llama_cpp",
    "torch": "torch",
    "torchvision": "torchvision",
    "sam2": "sam2",
}


def ensure_dependencies() -> None:
    missing = []
    for package, module_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package)

    if not missing:
        return

    requirements = Path(__file__).with_name("requirements.txt")
    if requirements.exists():
        print("Installing dependencies from requirements.txt …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        )
    else:
        print("Installing dependencies:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def main() -> None:
    ensure_dependencies()
    ensure_models()
    from smallville_gui import main as run_gui

    run_gui()


def ensure_models() -> None:
    """Install and verify the bundled local LLMs."""

    try:
        from littletown.ai.model_manager import ensure_local_models
    except Exception as exc:  # pragma: no cover - import guarding
        print(f"Skipping local model bootstrap: {exc}")
        return

    statuses = ensure_local_models(auto_download=True)
    for status in statuses:
        note = f" — {status.message}" if status.message else ""
        print(f"[models] {status.spec.name}: {status.state}{note}")


if __name__ == "__main__":
    main()

