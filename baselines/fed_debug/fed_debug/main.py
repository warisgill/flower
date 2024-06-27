"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import hydra
from omegaconf import DictConfig
from server import run_simulation


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline."""
    run_simulation(cfg)


if __name__ == "__main__":
    main()
