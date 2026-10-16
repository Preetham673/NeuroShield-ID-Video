"""
circuit_env.py

Shared helper for running Circom/SnarkJS commands.

Replaces the hardcoded C:\\Users\\HP\\... path that was duplicated in
generate_proof.py (both the Python path and the /mnt/c/... WSL path) and
verify_proof.py. On Windows the command runs through WSL with the path
converted automatically; on Linux/macOS it runs directly.
"""

import os
import shlex
import subprocess
from pathlib import Path, PureWindowsPath

# <root>/src/zkp/circuit_env.py -> <root>
PROJECT_DIR = Path(__file__).resolve().parents[2]

CIRCUITS = PROJECT_DIR / "circuits"

# Never let a stuck node/snarkjs process hang the whole app.
TIMEOUT_SECONDS = 120


def to_wsl_path(path):
    """C:\\Users\\HP\\x -> /mnt/c/Users/HP/x"""

    p = PureWindowsPath(path)

    drive = p.drive.rstrip(":").lower()

    return "/mnt/" + drive + "/" + "/".join(p.parts[1:])


def run_in_circuits(command):
    """
    Run a shell command inside the circuits folder.
    May raise subprocess.TimeoutExpired or OSError (e.g. WSL missing);
    callers handle those.
    """

    if os.name == "nt":

        cwd = to_wsl_path(CIRCUITS)

        args = ["wsl", "bash", "-c", f"cd {shlex.quote(cwd)} && {command}"]

    else:

        args = ["bash", "-c", f"cd {shlex.quote(str(CIRCUITS))} && {command}"]

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )