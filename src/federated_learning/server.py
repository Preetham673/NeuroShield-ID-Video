"""
server.py

Federated Learning Server
"""

import argparse

import flwr as fl
import torch

from src.deepfake_detection.model import build_model, MODEL_PATH

# Aggregated global model is written next to the centralized one.
FEDERATED_MODEL_PATH = MODEL_PATH.with_name("federated_global.pth")


def weighted_accuracy(metrics):
    """Average client accuracy, weighted by each client's example count."""

    total = sum(n for n, _ in metrics)

    if total == 0:
        return {}

    return {"accuracy": sum(n * m["accuracy"] for n, m in metrics) / total}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg that keeps the aggregated model.

    The original server ran the rounds and then threw the aggregated
    weights away, so federated training produced no usable model.
    """

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:

            arrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            keys = list(build_model(pretrained=False).state_dict().keys())

            state_dict = {k: torch.tensor(v) for k, v in zip(keys, arrays)}

            FEDERATED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save(state_dict, FEDERATED_MODEL_PATH)

            print(f"\nRound {server_round}: global model saved to {FEDERATED_MODEL_PATH}")

        return aggregated_parameters, metrics


def start_server(rounds=3, min_clients=2, address="127.0.0.1:8080"):

    print("\n=================================")
    print("Starting Federated Learning Server")
    print("=================================\n")

    strategy = SaveModelStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
    )

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(
            num_rounds=rounds
        ),
        strategy=strategy,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NeuroShield-ID federated server")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--address", default="127.0.0.1:8080")
    args = parser.parse_args()

    start_server(args.rounds, args.min_clients, args.address)