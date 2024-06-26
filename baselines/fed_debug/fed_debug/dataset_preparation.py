"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)

# if __name__ == "__main__":

#     download_and_preprocess()


import logging
from collections import Counter

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


def train_test_transforms_factory(cfg):
    """Create the train and test transforms for the dataset."""
    train_transforms = None
    test_transforms = None
    # image_processor = AutoImageProcessor.from_pretrained(cfg.mname)
    if cfg.dname == "cifar10":

        def apply_train_transformCifar(example):
            transform = Compose(
                [
                    Resize((32, 32)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["img"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["img"]
            # del example['label']
            return example

        def apply_test_transformCifar(example):
            transform = Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["img"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["img"]
            # del example['label']
            return example

        train_transforms = apply_train_transformCifar
        test_transforms = apply_test_transformCifar
    elif cfg.dname == "mnist":

        def apply_train_transformMnist(example):

            transform = Compose(
                [Resize((32, 32)), ToTensor(), Normalize((0.1307,), (0.3081,))]
            )
            # example['pixel_values'] = [
            #     transform(image.convert("RGB")) for image in example['image']]
            example["pixel_values"] = transform(example["image"].convert("RGB"))
            example["label"] = torch.tensor(example["label"])
            del example["image"]
            return example

        def apply_test_transform_mnist(example):
            # print(f"example : {example}")
            transform = Compose(
                [Resize((32, 32)), ToTensor(), Normalize((0.1307,), (0.3081,))]
            )

            # example['pixel_values'] = [
            #     transform(image.convert("RGB")) for image in example['image']]
            example["pixel_values"] = transform(example["image"].convert("RGB"))
            example["label"] = torch.tensor(example["label"])
            del example["image"]
            # del example['label']

            return example

        train_transforms = apply_train_transformMnist
        test_transforms = apply_test_transform_mnist

    else:
        raise ValueError(f"Unknown dataset: {cfg.dname}")

    return {"train": train_transforms, "test": test_transforms}


def getLabelsCount(partition, target_label_col):
    """Return the count of labels in the partition."""
    label2count = Counter(
        example[target_label_col] for example in partition  # type: ignore
    )  # type: ignore

    return dict(label2count)


def fixPartition(cfg, c_partition, target_label_col):
    """Fix the partition to have a minimum of 10 examples per class."""
    label2count = getLabelsCount(c_partition, target_label_col)

    filtered_labels = {
        label: count for label, count in label2count.items() if count >= 10
    }

    indices_to_select = [
        i
        for i, example in enumerate(c_partition)
        if example[target_label_col] in filtered_labels
    ]  # type: ignore

    ds = c_partition.select(indices_to_select)

    if len(ds) > cfg.max_per_client_data_size:
        # ds = ds.shuffle()
        ds = ds.select(range(cfg.max_per_client_data_size))

    if len(ds) % cfg.batch_size == 1:
        ds = ds.select(range(len(ds) - 1))

    partition_labels_count = getLabelsCount(ds, target_label_col)
    return {"partition": ds, "partition_labels_count": partition_labels_count}


def dirichletDataDistribution(
    cfg, target_label_col, fetch_only_test_data, subtask=None
):
    """Create a Dirichlet data distribution."""
    partitioner = DirichletPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        alpha=cfg.dirichlet_alpha,
        min_partition_size=0,
        self_balancing=True,
        shuffle=True,
    )

    # logging.info(f"Dataset name: {cfg.dname}")
    clients_class = []
    clients_data = []

    fds = None

    if subtask is not None:
        fds = FederatedDataset(
            dataset=cfg.dname, partitioners={"train": partitioner}, subset=subtask
        )
    else:
        fds = FederatedDataset(dataset=cfg.dname, partitioners={"train": partitioner})

    server_data = fds.load_split("test").select(range(cfg.max_server_data_size))

    logging.info(f"Server data keys {server_data[0].keys()}")

    if not fetch_only_test_data:
        for partition_index in range(cfg.num_clients):
            partition = fds.load_partition(partition_index)

            d = fixPartition(cfg, partition, target_label_col)

            if len(d["partition"]) >= cfg.batch_size:
                clients_data.append(d["partition"])
                clients_class.append(d["partition_labels_count"])

    # per client data size
    per_client_data_size = [len(dl) for dl in clients_data]
    logging.debug(
        f"Data per clients {per_client_data_size}, "
        f"server data size: {len(server_data)}, "
        f"fetch_only_test_data: {fetch_only_test_data}"
    )

    client2data = {f"{id}": v for id, v in enumerate(clients_data)}
    client2class = {f"{id}": v for id, v in enumerate(clients_class)}

    return {
        "client2data": client2data,
        "server_data": server_data,
        "client2class": client2class,
    }
