"""
client.py

Federated Learning Client
using NeuroShield-ID Trainer

Run one process per client, each with its own shard of the data:

    python -m src.federated_learning.client --client-id 0 --num-clients 2
    python -m src.federated_learning.client --client-id 1 --num-clients 2
"""

import argparse

import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

from src.deepfake_detection import trainer
from src.deepfake_detection.model import DEVICE
from src.deepfake_detection.trainer import (
    model,
    criterion,
    train_one_epoch,
)


def use_data_shard(client_id, num_clients):
    """
    Give this client its own slice of the training data.

    Previously every client trained on the full dataset, which is not
    federated learning (there was nothing distinct to federate). A strided
    split keeps each shard balanced between REAL and FAKE.
    """

    dataset = trainer.train_loader.dataset

    indices = list(range(client_id, len(dataset), num_clients))

    # trainer.train_one_epoch reads the module-level train_loader, so
    # replacing it here makes local training use only this shard.
    trainer.train_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=trainer.train_loader.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(f"Client {client_id}/{num_clients}: {len(indices)} training images")


def evaluate_local():
    """Return (cross-entropy loss, accuracy %) on the validation set."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in trainer.validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            total_loss += criterion(outputs, labels).item() * labels.size(0)

            correct += outputs.argmax(1).eq(labels).sum().item()

            total += labels.size(0)

    return total_loss / total, 100 * correct / total


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

        # Second value is the number of training EXAMPLES; FedAvg uses it to
        # weight this client. It used to be len(state_dict()), i.e. the
        # number of weight tensors, which is meaningless as a weight.
        return (
            self.get_parameters(config),
            len(trainer.train_loader.dataset),
            {}
        )

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        # Real validation loss (was a fake "1 - accuracy" stand-in).
        loss, accuracy = evaluate_local()

        print(f"\nValidation Accuracy: {accuracy:.2f}%")

        return (
            float(loss),
            len(trainer.validation_loader.dataset),
            {
                "accuracy": accuracy
            }
        )


def start_client(server_address="127.0.0.1:8080"):

    # start_numpy_client is deprecated in current Flower releases.
    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient().to_client(),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NeuroShield-ID federated client")
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--server", default="127.0.0.1:8080")
    args = parser.parse_args()

    if not 0 <= args.client_id < args.num_clients:
        parser.error("--client-id must be between 0 and --num-clients - 1")

    use_data_shard(args.client_id, args.num_clients)

    start_client(args.server)