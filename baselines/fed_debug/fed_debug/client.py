"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import logging
from collections import OrderedDict

import evaluate
import flwr as fl
import numpy as np
import torch
from transformers import TrainingArguments

from .models import CNNTrainer


def _compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


class CNNFlowerClient(fl.client.NumPyClient):
    """Flower client for training a CNN model."""

    def __init__(self, config):
        """Initialize the client with the given configuration."""
        self.config = config

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        nk_client_data_points = len(self.config["client_data_train"])
        local_epochs = config["local_epochs"]
        batch_size = config["batch_size"]
        lr = config["lr"]

        model_dict = self.config["model_dict"]
        client_net = model_dict["model"]

        client_net.zero_grad()

        logging.debug(
            f"> client {self.config['cid']} taining start, "
            f"local_epochs: {local_epochs}, "
            f"batch_size: {batch_size}, lr: {lr}, "
            f"nk_client_data_points: {nk_client_data_points}"
        )

        set_parameters(model=client_net, parameters=parameters)

        fp16 = True
        if self.config["device"].type == "cpu":
            fp16 = False

        training_args = TrainingArguments(
            output_dir=".exp_storage1/log_trainer",
            num_train_epochs=local_epochs,
            lr_scheduler_type="constant",
            eval_strategy="no",  # type: ignore
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=fp16,
            disable_tqdm=True,
            remove_unused_columns=False,
            learning_rate=lr,
            report_to="none",
        )  # type: ignore

        trainer = CNNTrainer(
            model=client_net,
            args=training_args,
            train_dataset=self.config["client_data_train"],
            compute_metrics=_compute_metrics,
            tokenizer=None,  # Not needed for CNNs
            data_collator=None,  # Not needed for CNNs
        )  # type: ignore

        trainer.train()

        client_net.eval()
        client_net = client_net.cpu()
        parameters = get_parameters(client_net)
        del trainer
        del client_net
        return parameters, nk_client_data_points, {"cid": self.config["cid"]}


def get_parameters(model):
    """Return model parameters as a list of NumPy ndarrays."""
    model = model.cpu()
    model.zero_grad()
    # Return all model parameters as a list of NumPy ndarrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    model.zero_grad()
    model = model.cpu()
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
