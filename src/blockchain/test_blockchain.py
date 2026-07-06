from src.blockchain.blockchain import Blockchain

blockchain = Blockchain()

blockchain.add_block(
    "Prediction=REAL, Trust=92.86"
)

blockchain.add_block(
    "Prediction=FAKE, Trust=15.23"
)

blockchain.display_chain()