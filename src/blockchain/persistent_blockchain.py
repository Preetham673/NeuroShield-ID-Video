"""
persistent_blockchain.py

Persistent Blockchain
Stores blockchain on disk.
"""

import json
import os
import time
import hashlib
import threading
from pathlib import Path


# Project root, derived from this file's location
# (<root>/src/blockchain/persistent_blockchain.py) instead of a
# hardcoded C:\Users\HP\... path that only worked on one machine.
PROJECT_DIR = Path(__file__).resolve().parents[2]

BLOCKCHAIN_FILE = (
    PROJECT_DIR /
    "outputs" /
    "blockchain" /
    "blockchain.json"
)

# Streamlit serves sessions from threads in one process, so guard
# read-modify-write cycles against each other.
_LOCK = threading.RLock()


class PersistentBlockchain:

    def __init__(self):

        with _LOCK:

            self.chain = self._load()

            if not self.chain:

                self.create_genesis_block()

                self.save()

    # ----------------------------

    def _load(self):

        if not BLOCKCHAIN_FILE.exists():
            return []

        try:
            with open(BLOCKCHAIN_FILE, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            # Don't crash the whole app on a damaged file; keep a copy
            # for inspection and start a fresh chain.
            backup = BLOCKCHAIN_FILE.with_suffix(
                f".corrupt-{int(time.time())}.json"
            )
            try:
                os.replace(BLOCKCHAIN_FILE, backup)
            except OSError:
                pass
            return []

    # ----------------------------

    def create_genesis_block(self):

        block = {

            "index": 0,

            "timestamp": time.time(),

            "data": "Genesis Block",

            "previous_hash": "0"

        }

        block["hash"] = self.calculate_hash(block)

        self.chain.append(block)

    # ----------------------------

    def calculate_hash(self, block):

        # Hash every field except the hash itself, so the same function
        # works both when creating a block and when validating one.
        payload = {k: v for k, v in block.items() if k != "hash"}

        block_string = json.dumps(
            payload,
            sort_keys=True
        ).encode()

        return hashlib.sha256(
            block_string
        ).hexdigest()

    # ----------------------------

    def add_block(self, data):

        with _LOCK:

            # Reload from disk first. Instances created earlier (e.g. at
            # import time) hold a stale in-memory chain; appending to it
            # would fork the chain and overwrite blocks added since.
            self.chain = self._load() or self.chain

            previous = self.chain[-1]

            block = {

                "index": len(self.chain),

                "timestamp": time.time(),

                "data": data,

                "previous_hash": previous["hash"]

            }

            block["hash"] = self.calculate_hash(block)

            self.chain.append(block)

            self.save()

            return block

    # ----------------------------

    def is_valid(self):

        """Return (ok, message) after checking hashes and links."""

        for i, block in enumerate(self.chain):

            if block["hash"] != self.calculate_hash(block):
                return False, f"Block #{block['index']} hash mismatch (tampered)."

            if i == 0:
                continue

            if block["previous_hash"] != self.chain[i - 1]["hash"]:
                return False, f"Block #{block['index']} link broken."

        return True, "Chain is valid."

    # ----------------------------

    def save(self):

        BLOCKCHAIN_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Write to a temp file then swap it in, so a crash mid-write
        # can't leave a half-written, unreadable blockchain.json.
        tmp = BLOCKCHAIN_FILE.with_suffix(".tmp")

        with open(tmp, "w") as f:

            json.dump(
                self.chain,
                f,
                indent=4
            )

        os.replace(tmp, BLOCKCHAIN_FILE)

    # ----------------------------

    def display(self):

        print()

        print("=" * 50)

        print("PERSISTENT BLOCKCHAIN")

        print("=" * 50)

        for block in self.chain:

            print()

            print(block)