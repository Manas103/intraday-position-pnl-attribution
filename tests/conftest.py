import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CPP_BINARY = ROOT / "cpp" / "build" / "attribution_engine"


def _build_cpp_engine():
    """Try to build the C++ engine via WSL if it is missing. If WSL is not
    available (e.g. CI on a non-Windows box without WSL), leave it missing
    and let dependent tests skip cleanly."""
    if CPP_BINARY.exists():
        return
    try:
        subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-22.04", "--", "bash", "-lc",
             "cd \"$(wslpath -a '" + str(ROOT / 'cpp').replace("'", "'\\''") + "')\" && make all"],
            capture_output=True, timeout=120, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


@pytest.fixture(scope="session")
def cpp_engine_path():
    _build_cpp_engine()
    if not CPP_BINARY.exists():
        pytest.skip("C++ attribution_engine binary not built and could not be built via WSL")
    return CPP_BINARY
