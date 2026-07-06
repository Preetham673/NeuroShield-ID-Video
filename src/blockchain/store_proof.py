import hashlib
from pathlib import Path

from src.blockchain.persistent_blockchain import PersistentBlockchain

PROJECT_DIR = Path(
    r"C:\Users\HP\OneDrive\Desktop\NeuroShield-ID"
)

CIRCUITS = PROJECT_DIR / "circuits"

blockchain = PersistentBlockchain()


def store_proof():

    proof_file = CIRCUITS / "proof.json"

    with open(proof_file, "rb") as f:

        proof_hash = hashlib.sha256(
            f.read()
        ).hexdigest()

    blockchain.add_block(proof_hash)

    print("\n========== BLOCKCHAIN ==========\n")

    print("Proof Hash Stored")

    print(proof_hash)

    return proof_hash