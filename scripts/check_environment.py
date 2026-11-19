"""Check workstation setup without importing unfinished exercise modules."""
from __future__ import annotations

import importlib
import platform
import shutil
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = []
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()} ({platform.machine()})")
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        errors.append("Use Python 3.12 for this course (supported: 3.11/3.12).")
    if not Path(sys.executable).resolve().is_relative_to((root / '.venv').resolve()):
        errors.append("Interpreter is outside the project .venv. Run with uv run --locked.")
    if Path.cwd().resolve() != root:
        errors.append(f"Run from the project root: {root}")
    for command in ('git', 'uv'):
        location = shutil.which(command)
        print(f"{command}: {location or 'MISSING'}")
        if not location:
            errors.append(f"Install {command} and reopen the terminal.")
    for name in ('torch', 'torchvision', 'mlflow', 'dvc', 'numpy', 'pandas',
                 'yaml', 'PIL', 'sklearn', 'matplotlib', 'pytest'):
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'import OK')}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        import torch
        result = torch.ones(2, 3) @ torch.ones(3, 2)
        assert result.shape == (2, 2) and result.sum().item() == 12
        print(f"CUDA available: {torch.cuda.is_available()} (False is expected for Core)")
        print('CPU tensor operation: PASS')
    except Exception as exc:
        errors.append(f"CPU tensor check: {exc}")
    for error in errors:
        print(f"FAIL: {error}")
    print('SETUP PASS' if not errors else 'SETUP FAIL - see docs/SETUP.md')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
