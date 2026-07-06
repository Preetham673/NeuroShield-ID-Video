import time

from src.blockchain.block import Block


class Blockchain:

    def __init__(self):

        self.chain = [
            self.create_genesis_block()
        ]

    def create_genesis_block(self):

        return Block(
            0,
            time.time(),
            "Genesis Block",
            "0"
        )

    def get_latest_block(self):

        return self.chain[-1]

    def add_block(self, data):

        latest = self.get_latest_block()

        new_block = Block(
            len(self.chain),
            time.time(),
            data,
            latest.hash
        )

        self.chain.append(new_block)

    def display_chain(self):

        for block in self.chain:

            print("\n========================")

            print("Index :", block.index)

            print("Time  :", block.timestamp)

            print("Data  :", block.data)

            print("Prev  :", block.previous_hash)

            print("Hash  :", block.hash)