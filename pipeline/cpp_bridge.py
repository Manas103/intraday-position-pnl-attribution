"""Subprocess bridge to the C++ revaluation/attribution engine.

Protocol (see cpp/src/attribution_main.cpp for the authoritative spec):
  stdin,  one line per position-session, CSV, no header:
    id,S0,K,T0,r,sigma0,type,S1,sigma1,T1
  stdout, one line per input line, same order, CSV, no header:
    id,price0,price1,actual_pnl,delta0,gamma0,vega0,theta0,
    delta_pnl,gamma_pnl,vega_pnl,theta_pnl,taylor_sum,residual
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BINARY = Path(__file__).resolve().parent.parent / "cpp" / "build" / "attribution_engine"


def _windows_path_to_wsl(path: Path) -> str:
    """Convert an absolute Windows path to the /mnt/<drive>/... form WSL
    expects. Used because the compiled attribution_engine is a Linux ELF
    binary (built by WSL2 g++) and cannot be exec'd directly by a
    Windows-native Python process: launching it with subprocess.run
    straight from Windows Python raises "OSError: [WinError 193] %1 is not
    a valid Win32 application". Routing the call through wsl.exe is what
    makes this bridge work from the Windows-side Python the rest of the
    pipeline runs under."""
    resolved = str(path.resolve())
    drive, rest = resolved.split(":", 1)
    return f"/mnt/{drive.lower()}{rest.replace(chr(92), '/')}"


def _build_command(binary: Path) -> list[str]:
    if sys.platform == "win32":
        return ["wsl.exe", "-d", "Ubuntu-22.04", "--", _windows_path_to_wsl(binary)]
    return [str(binary)]


@dataclass(frozen=True)
class AttributionResult:
    id: str
    price_sod: float
    price_eod: float
    actual_pnl: float
    delta: float
    gamma: float
    vega: float
    theta: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    taylor_sum: float
    residual: float


class EngineNotBuiltError(RuntimeError):
    pass


def run_attribution_batch(requests, binary_path: str | Path | None = None) -> list[AttributionResult]:
    """requests: iterable of tuples
    (id, S0, K, T0, r, sigma0, option_type, S1, sigma1, T1)."""
    binary = Path(binary_path) if binary_path else DEFAULT_BINARY
    if not binary.exists():
        raise EngineNotBuiltError(
            f"C++ engine not found at {binary}. Build it first: "
            f"wsl.exe -d Ubuntu-22.04 -- bash -lc 'cd cpp && make all'"
        )

    input_lines = [",".join(str(x) for x in req) for req in requests]
    stdin_data = "\n".join(input_lines) + ("\n" if input_lines else "")

    proc = subprocess.run(
        _build_command(binary), input=stdin_data, capture_output=True, text=True, check=True
    )

    results = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        p = line.split(",")
        results.append(
            AttributionResult(
                id=p[0],
                price_sod=float(p[1]),
                price_eod=float(p[2]),
                actual_pnl=float(p[3]),
                delta=float(p[4]),
                gamma=float(p[5]),
                vega=float(p[6]),
                theta=float(p[7]),
                delta_pnl=float(p[8]),
                gamma_pnl=float(p[9]),
                vega_pnl=float(p[10]),
                theta_pnl=float(p[11]),
                taylor_sum=float(p[12]),
                residual=float(p[13]),
            )
        )
    return results
