"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from .dataset_preparation import (
    dirichletDataDistribution,
    train_test_transforms_factory,
)


def initialize_image_dataset(cfg, fetch_only_test_data):
    """Initialize and return the image dataset."""
    target_label_col = "label"
    d = dirichletDataDistribution(cfg, target_label_col, fetch_only_test_data)
    transforms = train_test_transforms_factory(cfg=cfg)
    d["client2data"] = {
        k: v.map(transforms["train"]) for k, v in d["client2data"].items()
    }
    d["server_data"] = d["server_data"].map(transforms["test"])
    return d


def load_datasets(cfg, fetch_only_test_data=False):
    """Load the dataset and return the dataload."""
    if cfg.dname in ["cifar10", "mnist"]:
        return initialize_image_dataset(cfg, fetch_only_test_data)
    return None


def load_central_server_test_data(cfg):
    """Load the central server test data."""
    d = load_datasets(cfg, fetch_only_test_data=True)
    return d["server_data"]
