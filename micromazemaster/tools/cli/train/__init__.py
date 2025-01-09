import atexit
import random

import torch
import typer
from typing_extensions import Annotated

from micromazemaster.models.maze import Maze
from micromazemaster.utils.evolution import Evolution
from micromazemaster.utils.logging import logger
from micromazemaster.utils.preprocessing import create_tmp_dir

train_cli = typer.Typer(no_args_is_help=True)


@train_cli.callback()
def train_callback():
    """Micromazemaster module for training"""
    pass


@train_cli.command("evolution")
def cmd_train_evolution(
    num_generations: Annotated[int, typer.Option(help="Number of generations to train.")] = 2500,
    population_size: Annotated[int, typer.Option(help="Size of the population.")] = 25,
    mutation_rate: Annotated[float, typer.Option(help="The rate at which mutation occurs.")] = 0.1,
    crossover_rate: Annotated[float, typer.Option(help="The rate at which crossover occurs.")] = 0.5,
    starting_steps: Annotated[int, typer.Option(help="Number of steps to start with.")] = 10,
    parallel_execution: Annotated[bool, typer.Option(help="Whether to use parallel execution.")] = True,
    seed: Annotated[int, typer.Option(help="Seed for reproducable results.", show_default="random")] = random.randint(
        0, 100000
    ),
):
    """Train a model using the evolutionary algorithm"""

    WORKING_DIR = create_tmp_dir()

    def exit_handler():
        logger.info("Saving best model...")
        try:
            evo.save_as_state_dict(WORKING_DIR.joinpath("best_model.pth"))
            logger.info("Best model saved.")
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
        logger.info("Exiting...")

    atexit.register(exit_handler)

    random.seed(seed)

    torch.manual_seed(seed)

    seeds = [random.randint(0, 100000) for _ in range(31)]

    mazes = [Maze(width=5, height=5, seed=seed) for seed in seeds]

    train_mazes = mazes[:-1]

    device = torch.device("cpu")
    evo = Evolution(
        num_generations=num_generations,
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        device=device,
    )
    best_model = evo.train(
        train_mazes=train_mazes, starting_steps=starting_steps, parallel_execution=parallel_execution
    )
    logger.debug(f"Best model: {best_model.state_dict()}")
