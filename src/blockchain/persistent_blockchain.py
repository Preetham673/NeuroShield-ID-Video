"""
persistent_blockchain.py

Persistent Blockchain
Stores blockchain on disk.
"""

import json
import time
import hashlib
from pathlib import Path


PROJECT_DIR = Path(
    r"C:\Users\HP\OneDrive\Desktop\NeuroShield-ID"
)

BLOCKCHAIN_FILE = (
    PROJECT_DIR /
    "outputs" /
    "blockchain" /
    "blockchain.json"
)


class PersistentBlockchain:

    def __init__(self):

        if BLOCKCHAIN_FILE.exists():

            with open(BLOCKCHAIN_FILE, "r") as f:

                self.chain = json.load(f)

        else:

            self.chain = []

            self.create_genesis_block()

            self.save()

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

        block_string = json.dumps(
            block,
            sort_keys=True
        ).encode()

        return hashlib.sha256(
            block_string
        ).hexdigest()

    # ----------------------------

    def add_block(self, data):

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

    # ----------------------------

    def save(self):

        BLOCKCHAIN_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            BLOCKCHAIN_FILE,
            "w"
        ) as f:

            json.dump(
                self.chain,
                f,
                indent=4
            )

    # ----------------------------

    def display(self):

        print()

        print("=" * 50)

        print("PERSISTENT BLOCKCHAIN")

        print("=" * 50)

        for block in self.chain:

            print()

            print(block)