import hashlib
from pathlib import Path

from src.blockchain.persistent_blockchain import PersistentBlockchain

# <root>/src/blockchain/store_proof.py -> <root>
PROJECT_DIR = Path(__file__).resolve().parents[2]

CIRCUITS = PROJECT_DIR / "circuits"


def store_proof():

    proof_file = CIRCUITS / "proof.json"

    with open(proof_file, "rb") as f:

        proof_hash = hashlib.sha256(
            f.read()
        ).hexdigest()

    # Created per call (not at import) so it always starts from the
    # chain currently on disk instead of a stale import-time copy.
    blockchain = PersistentBlockchain()

    blockchain.add_block(proof_hash)

    print("\n========== BLOCKCHAIN ==========\n")

    print("Proof Hash Stored")

    print(proof_hash)

    return proof_hash