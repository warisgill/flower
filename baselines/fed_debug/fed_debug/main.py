"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import hydra
from omegaconf import DictConfig
from server import run_simulation
import copy
import logging


def fed_debug_multiple_faulty_clients_experiment(main_cfg):
    """Table 2 of the fed_debug paper."""
    faulty_clients_nums = [2,3,5,7]
    mnames  = ['resnet50', 'densenet121']
    total_clients = [30, 50]
    # 'flwrlabs/femnist'
    dataset_list = ['cifar10']
    all_configs = []
    for dname in dataset_list:
        for mname in mnames:
            for n_cs in total_clients:
                for f_clients_nums in faulty_clients_nums:
                    cfg = copy.deepcopy(main_cfg)
                    cfg.dataset.name = dname
                    cfg.model.name = mname
                    cfg.strategy.clients_per_round = n_cs
                    cfg.data_dist.num_clients = n_cs
                    cfg.data_dist.dname = dname
                    cfg.faulty_clients_ids = [f"{i}" for i in range(f_clients_nums)]
                    all_configs.append(cfg)
                    print(">> ---> cfg", cfg)

    for cfg in all_configs:
        logging.info(f'>>> Running simulation for {cfg}')
        run_simulation(cfg)




@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline."""
    run_simulation(cfg)
    # fed_debug_multiple_faulty_clients_experiment(cfg)


if __name__ == "__main__":
    main()
