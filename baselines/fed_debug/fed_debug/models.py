"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import logging

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18
from transformers import DefaultDataCollator, Trainer, TrainingArguments


def _compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    correct_predictions = predictions == labels
    correct_indices = np.where(correct_predictions)[0]
    correct_indices = torch.from_numpy(correct_indices)
    d = {
        "accuracy": metric.compute(predictions=predictions, references=labels),
        "correct_indices": correct_indices,
        "actual_labels": labels,
    }
    return d


def initialize_model(name, cfg_dataset):
    """Initialize the model with the given name."""
    model_dict = {"model": None}
    if name in ["resnet18"]:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, cfg_dataset.num_classes)
        model_dict["model"] = model.cpu()
    else:
        raise ValueError(f"Model {name} not supported")

    return model_dict


class LeNet(nn.Module):
    """LeNet model for image classification.

    This class implements the LeNet architecture for image classification, consisting of
    convolutional and fully connected layers.
    """

    def __init__(self, cfg_dataset) -> None:
        """Initialize the LeNet model."""
        super().__init__()
        self.conv1 = nn.Conv2d(cfg_dataset.channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, cfg_dataset.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNTrainer(Trainer):
    """Trainer for the CNN model."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss on the given model."""
        labels = inputs.get("labels")
        batch_inputs = inputs.get("pixel_values")
        outputs = model(batch_inputs)
        logits = outputs
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Perform an evaluation step on the given model."""
        loss = None
        labels = None
        logits = None
        with torch.no_grad():
            logits = model(inputs["pixel_values"])
            if "labels" in inputs:
                labels = inputs.get("labels")
                loss = nn.CrossEntropyLoss()(logits, labels)

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def _test_cnn_model(gm_dict, central_server_test_data, batch_size):
    net = gm_dict["model"]
    logging.debug("Evaluating cnn model")
    testing_args = TrainingArguments(
        logging_strategy="steps",
        output_dir=".exp_storage1/log_trainer",
        do_train=False,  # Disable training
        do_eval=True,  # Enable evaluation
        per_device_eval_batch_size=batch_size,
        disable_tqdm=True,  # Enable tqdm progress bar
        remove_unused_columns=False,
        report_to="none",
    )

    tester = CNNTrainer(
        model=net,
        args=testing_args,
        # Ensure it uses the correct metrics for evaluation
        compute_metrics=_compute_metrics,
        data_collator=DefaultDataCollator(),
    )

    logging.debug(f"lenght of eval dataset: {len(central_server_test_data)}")
    # Evaluate the model on the test dataset
    r = tester.evaluate(eval_dataset=central_server_test_data)
    r["eval_accuracy"] = r["eval_accuracy"]["accuracy"]  # type: ignore
    net = net.cpu()
    return r


def global_model_eval(arch, global_net_dict, server_testdata, batch_size=16):
    """Evaluate the global model on the server test data."""
    d = {}
    if arch == "cnn":
        d = _test_cnn_model(
            global_net_dict,
            central_server_test_data=server_testdata,
            batch_size=batch_size,
        )
    return {
        "loss": d["eval_loss"],
        "accuracy": d["eval_accuracy"],
        "eval_correct_indices": d["eval_correct_indices"],
        "eval_actual_labels": d["eval_actual_labels"],
    }
