"""
server.py

Federated Learning Server
"""

import flwr as fl


def start_server():

    print("\n=================================")
    print("Starting Federated Learning Server")
    print("=================================\n")

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(
            num_rounds=3
        ),
    )


if __name__ == "__main__":
    start_server()