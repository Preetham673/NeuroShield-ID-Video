"""
generate_proof.py

Generate zk-SNARK proof
from a Trust Score.
"""

import json
import subprocess
from pathlib import Path


PROJECT_DIR = Path(
    r"C:\Users\HP\OneDrive\Desktop\NeuroShield-ID"
)

CIRCUITS = PROJECT_DIR / "circuits"


def generate_proof(trust_score):

    # -----------------------------------
    # Create input.json automatically
    # -----------------------------------

    data = {
        "trustScore": int(trust_score)
    }

    with open(
        CIRCUITS / "input.json",
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    # -----------------------------------
    # Generate witness + proof
    # -----------------------------------

    command = (
        'wsl bash -c '
        '"cd /mnt/c/Users/HP/OneDrive/Desktop/NeuroShield-ID/circuits && '
        'node trust_score_js/generate_witness.js '
        'trust_score_js/trust_score.wasm '
        'input.json '
        'witness.wtns && '
        'snarkjs groth16 prove '
        'trust_score_final.zkey '
        'witness.wtns '
        'proof.json '
        'public.json"'
    )

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print(result.stderr)

        return False

    print("\nProof Generated Successfully")

    return True