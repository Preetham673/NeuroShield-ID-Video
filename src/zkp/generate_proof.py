"""
generate_proof.py

Generate zk-SNARK proof
from a Trust Score.
"""

import json
import subprocess

from src.zkp.circuit_env import CIRCUITS, run_in_circuits


def generate_proof(trust_score):

    # Remove leftovers so a failed run can never leave an old proof
    # behind to be verified/stored as if it were fresh.
    for name in ("proof.json", "public.json", "witness.wtns"):
        (CIRCUITS / name).unlink(missing_ok=True)

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
        "node trust_score_js/generate_witness.js "
        "trust_score_js/trust_score.wasm "
        "input.json "
        "witness.wtns && "
        "snarkjs groth16 prove "
        "trust_score_final.zkey "
        "witness.wtns "
        "proof.json "
        "public.json"
    )

    try:

        result = run_in_circuits(command)

    except (subprocess.TimeoutExpired, OSError) as e:

        print(f"Proof generation could not run: {e}")

        return False

    if result.returncode != 0:

        print(result.stderr)

        return False

    print("\nProof Generated Successfully")

    return True