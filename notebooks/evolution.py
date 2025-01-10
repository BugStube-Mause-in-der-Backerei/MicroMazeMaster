# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import atexit
import os
import random
from datetime import datetime
from pathlib import Path

import torch

from micromazemaster.models.maze import Maze
from micromazemaster.utils.evolution import Evolution
from micromazemaster.utils.logging import logger

# +
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
CWD = Path(os.getcwd())
WORKING_DIR = CWD.joinpath("local_data", timestamp)
try:
    WORKING_DIR.mkdir(parents=True)
    logger.debug(f"Working directory created: {WORKING_DIR}")

except OSError as e:
    logger.error(f"Failed to create working directory: {e}")
    exit(1)


# +
def exit_handler():
    logger.info("Saving best model...")
    try:
        evo.save_as_state_dict(WORKING_DIR.joinpath("best_model.pth"))
        logger.info("Best model saved.")
    except Exception as e:
        logger.error(f"Failed to save best model: {e}")
    logger.info("Exiting...")


atexit.register(exit_handler)

# +
random.seed(42)

torch.manual_seed(42)

seeds = [random.randint(0, 100000) for _ in range(31)]

# Generate a maze
mazes = [Maze(width=5, height=5, seed=seed) for seed in seeds]

# Split the mazes into training and evaluation
train_mazes = mazes[:-1]


# +
device = torch.device("cpu")
evo = Evolution(num_generations=2500, population_size=25, mutation_rate=0.1, crossover_rate=0.5, device=device)
best_model = evo.train(train_mazes=train_mazes, starting_steps=0, parallel_execution=True)
logger.debug(f"Best model: {best_model.state_dict()}")
