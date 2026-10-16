"""
verify_proof.py

Verify zk-SNARK proof and
check policy result.
"""

import json
import subprocess

from src.zkp.circuit_env import CIRCUITS, run_in_circuits

_FAILED = {
    "proof_verified": False,
    "policy_passed": False
}


def verify_proof():

    command = (
        "snarkjs groth16 verify "
        "verification_key.json "
        "public.json "
        "proof.json"
    )

    try:

        result = run_in_circuits(command)

    except (subprocess.TimeoutExpired, OSError) as e:

        print(f"Proof verification could not run: {e}")

        return dict(_FAILED)

    if "OK!" not in result.stdout:

        print(result.stderr)

        return dict(_FAILED)

    # -----------------------------
    # Read circuit output
    # -----------------------------

    try:

        with open(
            CIRCUITS / "public.json",
            "r"
        ) as f:

            public_output = json.load(f)

        valid = int(public_output[0])

    except (OSError, ValueError, IndexError, TypeError) as e:

        print(f"Could not read circuit output: {e}")

        return dict(_FAILED)

    return {

        "proof_verified": True,

        "policy_passed": valid == 1

    }