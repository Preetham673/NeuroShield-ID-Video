"""
client.py

Federated Learning Client
using NeuroShield-ID Trainer
"""

import flwr as fl
import torch

from src.deepfake_detection.trainer import (
    model,
    train_one_epoch,
    validate,
)


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return [
            val.cpu().numpy()
            for _, val in model.state_dict().items()
        ]

    def set_parameters(self, parameters):

        params_dict = zip(
            model.state_dict().keys(),
            parameters
        )

        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        model.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        print("\nLocal Training Started...\n")

        train_one_epoch(1)

        print("\nLocal Training Finished\n")

        return (
            self.get_parameters(config),
            len(model.state_dict()),
            {}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        accuracy = validate()

        loss = 1 - (accuracy / 100)

        print(f"\nValidation Accuracy: {accuracy:.2f}%")

        return (
            loss,
            len(model.state_dict()),
            {
                "accuracy": accuracy
            }
        )


def start_client():

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )


if __name__ == "__main__":
    start_client()