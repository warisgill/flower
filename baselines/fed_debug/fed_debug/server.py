"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

import gc
import logging
import random
import time

import flwr as fl
import torch
from diskcache import Index
from flwr.common import ndarrays_to_parameters

from .client import CNNFlowerClient, get_parameters, set_parameters
from .dataset import load_datasets
from .models import globalModelEval, initializeModel
from .strategy import SaveModelStrategy


class FLSimulation:
    """Main class to run the simulation."""

    def __init__(self, cfg, cache):
        self.all_rounds_results = []
        self.cache = cache
        self.cfg = cfg
        self.strategy = None
        self.client_resources = {"num_cpus": cfg.client_cpus}
        if self.cfg.device == "cuda":
            self.client_resources = {
                "num_gpus": cfg.client_gpu,
                "num_cpus": cfg.client_cpus,
            }

        # if self.cfg.model.name == "LeNet":
        #     self.DEVICE = torch.device("cpu")
        #     logging.info("LeNet is running on CPU")
        #     self.client_resources = {"num_cpus": cfg.client_cpus}

        init_args = {"num_cpus": self.cfg.total_cpus, "num_gpus": self.cfg.total_gpus}
        self.backend_config = {
            "client_resources": self.client_resources,
            "init_args": init_args,
        }
        self._setup()

    def _setup(self):
        d = load_datasets(self.cfg.data_dist)
        self.trainloaders = d["client2data"]
        self.server_testdata = d["server_data"]
        self.client2class = d["client2class"]

        # logging.info(f"> Noisy clients: {self.noisy_clients}")
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
        self._setStrategy()

    def _setStrategy(self):
        initial_net = initializeModel(self.cfg.model.name, self.cfg.dataset)["model"]
        if self.cfg.strategy.name in ["fedavg", "fedprox"]:
            strategy = SaveModelStrategy(
                fraction_fit=0,  # -------> Fix
                fraction_evaluate=0.0,
                min_fit_clients=self.cfg.strategy.clients_per_round,
                min_evaluate_clients=0,
                min_available_clients=self.cfg.data_dist.num_clients,
                initial_parameters=ndarrays_to_parameters(
                    ndarrays=get_parameters(initial_net)
                ),
                evaluate_fn=self._evaluateGlobalModel,  # ignore
                on_fit_config_fn=self._getFit_Config,  # Pass the fit_config function
            )
            strategy.set_cache_and_exp_key(self.cache, self.cfg)
            self.strategy = strategy

    def _getFit_Config(self, server_round: int):
        # vERY IMPORTANT otherwise same clients are selected in each round
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

    def _evaluateGlobalModel(self, server_round, parameters):
        gm_dict = initializeModel(self.cfg.model.name, self.cfg.dataset)
        set_parameters(gm_dict["model"], parameters)
        gm_dict["model"].eval()  # type: ignore
        d = globalModelEval(self.cfg.model.arch, gm_dict, self.server_testdata)
        loss = d["loss"]
        accuracy = d["accuracy"]
        self.all_rounds_results.append({"loss": loss, "accuracy": accuracy})
        return loss, {"accuracy": accuracy, "loss": loss, "round": server_round}

    def _getClient(self, cid):
        model_dict = initializeModel(self.cfg.model.name, self.cfg.dataset)
        client = None
        print(f"-----> Client {cid} is being initialized")
        args = {
            "cid": cid,
            "model_dict": model_dict,
            "client_data_train": self.trainloaders[cid],
            "valloader": None,
            "device": self.DEVICE,
            "mode": self.cfg.strategy.name,
        }

        client = CNNFlowerClient(args).to_client()
        return client

    def run(self):
        """Run the simulation."""
        client_app = fl.client.ClientApp(client_fn=self._getClient)

        server_config = fl.server.ServerConfig(num_rounds=self.cfg.strategy.num_rounds)
        server_app = fl.server.ServerApp(config=server_config, strategy=self.strategy)

        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=self.cfg.data_dist.num_clients,
            backend_config=self.backend_config,
        )  # type: ignore

        return self.all_rounds_results


def runSimulation(cfg):
    """Run the simulation."""

    def setExpKey(cfg):
        key = (
            f"{cfg.model.name}-{cfg.dataset.name}-dp[{cfg.strategy.noise_multiplier}+"
            f"{cfg.strategy.clipping_norm}]-TClients{cfg.data_dist.num_clients}-"
            f"{cfg.strategy.name}-(R{cfg.strategy.num_rounds}"
            f"-clientsPerR{cfg.strategy.clients_per_round})"
            f"-{cfg.data_dist.dist_type}{cfg.data_dist.dirichlet_alpha}"
            f"-batch{cfg.data_dist.batch_size}-epochs{cfg.client.epochs}-"
            f"lr{cfg.client.lr}"
        )
        return key

    # Set minimum and maximum sleep durations in seconds
    min_sleep = 0.0  # Minimum sleep time in seconds
    max_sleep = 10.0  # Maximum sleep time in seconds

    # Generate a random sleep duration
    sleep_duration = random.uniform(min_sleep, max_sleep)

    logging.info(f"Sleeping for {sleep_duration} seconds")

    # Sleep for the random duration
    time.sleep(sleep_duration)

    cache = Index(cfg.storage.dir + cfg.storage.cache_name)

    exp_key = setExpKey(cfg)
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

    start_time = time.time()

    sim = FLSimulation(cfg, cache)
    round2results = sim.run()

    end_time = time.time()

    avg_sim_time_per_round = (end_time - start_time) / cfg.strategy.num_rounds

    cache[cfg.key] = {
        "client2class": sim.client2class,
        "train_cfg": cfg,
        "complete": True,
        "all_ronuds_gm_results": round2results,
        "avg_sim_time_per_round": avg_sim_time_per_round,
    }

    logging.info(f"Results of gm evaluations each round: {round2results}")
    logging.info(
        f"Simulation Complete for: {cfg.key} "
        f"and avg time per round: {avg_sim_time_per_round} seconds"
    )
