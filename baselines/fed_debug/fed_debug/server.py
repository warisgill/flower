"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

import gc
import logging
import random

import flwr as fl
import torch
from client import CNNFlowerClient, get_parameters, set_parameters
from dataset import load_datasets
from diskcache import Index
from flwr.common import ndarrays_to_parameters
from models import global_model_eval, initialize_model
from strategy import FedAvgSave
from utils import add_noise_in_data
import torch


class FLSimulation:
    """Main class to run the simulation."""

    def __init__(self, cfg, cache):
        self.all_rounds_results = []
        self.cache = cache
        self.cfg = cfg
        self.strategy = None
        self.device = torch.device(self.cfg.device)

        self.client_resources = {"num_cpus": cfg.client_cpus}
        if self.cfg.device == "cuda":
            self.client_resources = {
                "num_gpus": cfg.client_gpu,
                "num_cpus": cfg.client_cpus,
            }

        init_args = {"num_cpus": self.cfg.total_cpus, "num_gpus": self.cfg.total_gpus}
        self.backend_config = {
            "client_resources": self.client_resources,
            "init_args": init_args,
        }
        self._setup()

    def make_faulty_clients(self):
        """Make clients faulty."""
        for cid in self.cfg.faulty_clients_ids:
            self.trainloaders[cid] = add_noise_in_data(
                client_data=self.trainloaders[cid],
                label_col="label",
                noise_rate=self.cfg.noise_rate,
            )
            logging.warning(f"Client {cid} is made noisy \n  ")
            self.client2class[cid] = "noisy"

    def _setup(self):
        d = load_datasets(self.cfg.data_dist)
        self.trainloaders = d["client2data"]

        self.server_testdata = d["server_data"]
        self.client2class = d["client2class"]
        print(f'client2class: {self.client2class}')

        if len(self.cfg.faulty_clients_ids) > 0:
            self.make_faulty_clients()

        logging.info(f"> client2class {self.client2class}")

        if len(self.trainloaders) < self.cfg.data_dist.num_clients:
            logging.warning(
                f"orignal number of clients {self.cfg.data_dist.num_clients} "
                f"reduced to {len(self.trainloaders)}"
            )
            self.cfg.data_dist.num_clients = len(self.trainloaders)

        data_per_client = [len(dl) for dl in self.trainloaders.values()]
        logging.info(f"Data per client in experiment {data_per_client}")

        # min_data = min([len(dl) for dl in self.trainloaders.values()])
        min_data = min(len(dl) for dl in self.trainloaders.values())

        logging.info(f"Min data on a client: {min_data}")
        self._set_strategy()

    def _set_strategy(self):
        initial_net = initialize_model(self.cfg.model.name, self.cfg.dataset)["model"]
        if self.cfg.strategy.name in ["fedavg", "fedprox"]:
            strategy = FedAvgSave(
                cfg=self.cfg,
                cache=self.cache,
                fraction_fit=0,  # -------> Fix
                fraction_evaluate=0.0,
                min_fit_clients=self.cfg.strategy.clients_per_round,
                min_evaluate_clients=0,
                min_available_clients=self.cfg.data_dist.num_clients,
                initial_parameters=ndarrays_to_parameters(
                    ndarrays=get_parameters(initial_net)
                ),
                evaluate_fn=self._evaluate_global_model,  # ignore
                on_fit_config_fn=self._get_fit_config,  # Pass the fit_config function
                fit_metrics_aggregation_fn= self._fit_metrics_aggregation_fn,
            )
            self.strategy = strategy
    def _fit_metrics_aggregation_fn(self, metrics):
        """Aggregate metrics."""
        # loss = sum(m["loss"] for m in metrics) / len(metrics)
        # accuracy = sum(m["accuracy"] for m in metrics) / len(metrics)
        print(f"metrics: {metrics}")
        return {"loss": 0.1, "accuracy": 0.2}

    def _get_fit_config(self, server_round: int):
        random.seed(server_round)
        torch.manual_seed(server_round)
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": self.cfg.client.epochs,  #
            "batch_size": self.cfg.data_dist.batch_size,
            "lr": self.cfg.client.lr,
        }
        gc.collect()
        return config

    def _evaluate_global_model(self, server_round, parameters, config):
        gm_dict = initialize_model(self.cfg.model.name, self.cfg.dataset)
        set_parameters(gm_dict["model"], parameters)
        gm_dict["model"].eval()  # type: ignore
        d = global_model_eval(self.cfg.model.arch, gm_dict, self.server_testdata)
        loss = d["loss"]
        accuracy = d["accuracy"]
        self.all_rounds_results.append({"loss": loss, "accuracy": accuracy})
        return loss, {"accuracy": accuracy, "loss": loss, "round": server_round}

    def _get_client(self, cid):
        model_dict = initialize_model(self.cfg.model.name, self.cfg.dataset)
        client = None
        args = {
            "cid": cid,
            "model_dict": model_dict,
            "client_data_train": self.trainloaders[cid],
            "valloader": None,
            "device": self.device,
            "mode": self.cfg.strategy.name,
        }

        client = CNNFlowerClient(args).to_client()
        return client

    def run(self):
        """Run the simulation."""
        client_app = fl.client.ClientApp(client_fn=self._get_client)

        server_config = fl.server.ServerConfig(num_rounds=self.cfg.strategy.num_rounds)
        server_app = fl.server.ServerApp(config=server_config, strategy=self.strategy)

        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=self.cfg.data_dist.num_clients,
            backend_config=self.backend_config,  # type: ignore
        )

        return self.all_rounds_results


def run_simulation(cfg):
    """Run the simulation."""

    def set_exp_key(cfg):
        key = (
            f"{cfg.model.name}-{cfg.dataset.name}-"
            f"faulty_clients[{cfg.faulty_clients_ids}]-"
            f"noise_rate{cfg.noise_rate}-"
            f"TClients{cfg.data_dist.num_clients}-"
            f"{cfg.strategy.name}-(R{cfg.strategy.num_rounds}"
            f"-clientsPerR{cfg.strategy.clients_per_round})"
            f"-{cfg.data_dist.dist_type}{cfg.data_dist.dirichlet_alpha}"
            f"-batch{cfg.data_dist.batch_size}-epochs{cfg.client.epochs}-"
            f"lr{cfg.client.lr}"
        )
        return key

    cache = Index(cfg.storage.dir + cfg.storage.cache_name)

    exp_key = set_exp_key(cfg)
    cfg.key = exp_key
    logging.info(
        f" ********************  Starting Experiment: {cfg.key} ********************"
    )

    if cfg.check_cache:
        if cfg.key in cache:
            temp_dict = cache[cfg.key]
            # type: ignore
            if "complete" in temp_dict and temp_dict["complete"]:  # type: ignore
                logging.info(f"Experiment already completed: {cfg.key}")
                return

    logging.info(f"Simulation Configuration: {cfg}")

    sim = FLSimulation(cfg, cache)
    round2results = sim.run()


    temp_input = torch.tensor(sim.server_testdata[0]["pixel_values"])


    cache[cfg.key] = {
        "client2class": sim.client2class,
        "train_cfg": cfg,
        "complete": True,
        'input_shape': temp_input.shape,
        "all_ronuds_gm_results": round2results,
    }

    logging.info(f"Results of gm evaluations each round: {round2results}")
    logging.info(f"Simulation Complete for: {cfg.key} ")
