"""
verify_proof.py

Verify zk-SNARK proof and
check policy result.
"""

import json
import subprocess
from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\HP\OneDrive\Desktop\NeuroShield-ID"
)

CIRCUITS = PROJECT_DIR / "circuits"


def verify_proof():

    command = (
        'wsl bash -c '
        '"cd /mnt/c/Users/HP/OneDrive/Desktop/NeuroShield-ID/circuits && '
        'snarkjs groth16 verify '
        'verification_key.json '
        'public.json '
        'proof.json"'
    )

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    if "OK!" not in result.stdout:

        print(result.stderr)

        return {
            "proof_verified": False,
            "policy_passed": False
        }

    # -----------------------------
    # Read circuit output
    # -----------------------------

    with open(
        CIRCUITS / "public.json",
        "r"
    ) as f:

        public_output = json.load(f)

    valid = int(public_output[0])

    return {

        "proof_verified": True,

        "policy_passed": valid == 1

    }