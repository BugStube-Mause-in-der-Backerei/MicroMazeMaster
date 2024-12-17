import concurrent.futures
import os
import random
from enum import IntEnum
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from micromazemaster.models.maze import Maze, Orientation
from micromazemaster.models.mouse import Mouse
from micromazemaster.utils.config import settings
from micromazemaster.utils.logging import logger


class Action(IntEnum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2


def detect_walls(mouse: Mouse, maze: Maze) -> tuple[bool, bool, bool]:
    """Detects walls around the mouse.

    Args:
        - mouse (Mouse): The mouse object.
        - maze (Maze): The maze object.

    Returns:
        - tuple[bool, bool, bool]: The presence of walls in the left, front, and right directions.
    """
    wall_left = False
    wall_front = False
    wall_right = False

    match mouse.orientation:
        case Orientation.NORTH:
            wall_left = (mouse.position[0] - 1, mouse.position[1]) not in nx.neighbors(maze.graph, mouse.position)
            wall_front = (mouse.position[0], mouse.position[1] + 1) not in nx.neighbors(maze.graph, mouse.position)
            wall_right = (mouse.position[0] + 1, mouse.position[1]) not in nx.neighbors(maze.graph, mouse.position)
        case Orientation.EAST:
            wall_left = (mouse.position[0], mouse.position[1] + 1) not in nx.neighbors(maze.graph, mouse.position)
            wall_front = (mouse.position[0] + 1, mouse.position[1]) not in nx.neighbors(maze.graph, mouse.position)
            wall_right = (mouse.position[0], mouse.position[1] - 1) not in nx.neighbors(maze.graph, mouse.position)
        case Orientation.SOUTH:
            wall_left = (mouse.position[0] + 1, mouse.position[1]) not in nx.neighbors(maze.graph, mouse.position)
            wall_front = (mouse.position[0], mouse.position[1] - 1) not in nx.neighbors(maze.graph, mouse.position)
            wall_right = (mouse.position[0] - 1, mouse.position[1]) not in nx.neighbors(maze.graph, mouse.position)
        case Orientation.WEST:
            wall_left = (mouse.position[0], mouse.position[1] - 1) not in nx.neighbors(maze.graph, mouse.position)
            wall_front = (mouse.position[0] - 1, mouse.position[1]) not in nx.neighbors(maze.graph, mouse.position)
            wall_right = (mouse.position[0], mouse.position[1] + 1) not in nx.neighbors(maze.graph, mouse.position)

    return wall_left, wall_front, wall_right


class MazeSolver(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MazeSolver, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Evolution:
    def __init__(
        self,
        num_generations: int,
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        device: torch.device,
    ):
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.device = device
        self.best_model = None
        self.population = [
            MazeSolver(settings.evolution.input_size, settings.evolution.hidden_size, settings.evolution.output_size)
            for _ in range(population_size)
        ]

    def mutate(self, model: nn.Module) -> nn.Module:
        """Mutates the model by adding Gaussian noise to the parameters with a probability of mutation_rate.

        Args:
            - model (nn.Module): The model to be mutated.
            - mutation_rate (float): The probability of mutation.

        Returns:
            - nn.Module: The mutated model.
        """
        for param in model.parameters():
            if torch.rand(1).item() < self.mutation_rate:
                param.data += torch.randn_like(param.data) * 0.1  # Adding Gaussian noise with std=0.1
        return model

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> tuple[nn.Module, nn.Module]:
        """Crossover between two models.

        Args:
            - parent1 (nn.Module): The first parent model.
            - parent2 (nn.Module): The second parent model.

        Returns:
            - tuple[nn.Module, nn.Module]: The two offspring models.
        """
        child1 = MazeSolver(
            settings.evolution.input_size, settings.evolution.hidden_size, settings.evolution.output_size
        )
        child2 = MazeSolver(
            settings.evolution.input_size, settings.evolution.hidden_size, settings.evolution.output_size
        )

        for child1_layer, child2_layer, parent1_layer, parent2_layer in zip(
            child1.children(), child2.children(), parent1.children(), parent2.children()
        ):
            if isinstance(parent1_layer, torch.nn.Linear):
                # Single-point crossover for Linear layers
                crossover_point = random.randint(1, parent1_layer.weight.data.size(0) - 1)

                child1_layer.weight.data = torch.cat(
                    (parent1_layer.weight.data[:crossover_point], parent2_layer.weight.data[crossover_point:]), dim=0
                )
                child2_layer.weight.data = torch.cat(
                    (parent2_layer.weight.data[:crossover_point], parent1_layer.weight.data[crossover_point:]), dim=0
                )

                # Bias crossover (if applicable)
                if parent1_layer.bias is not None:
                    crossover_point = random.randint(1, parent1_layer.bias.data.size(0) - 1)
                    child1_layer.bias.data = torch.cat(
                        (parent1_layer.bias.data[:crossover_point], parent2_layer.bias.data[crossover_point:]), dim=0
                    )
                    child2_layer.bias.data = torch.cat(
                        (parent2_layer.bias.data[:crossover_point], parent1_layer.bias.data[crossover_point:]), dim=0
                    )

        return child1, child2

    def evaluate_individual(
        self, individual: nn.Module, train_mazes: list[Maze], steps: int, device: torch.device
    ) -> dict:
        """Evaluates the individual on the given mazes.

        Args:
            - individual (nn.Module): The individual to evaluate.
            - train_mazes (list[Maze]): The list of mazes to evaluate the individual on.
            - steps (int): The number of steps to evaluate the individual on each maze.
            - device (torch.device): The device to run the evaluation on.

        Returns:
            - dict: The evaluation results.
                    {
                        "individual": nn.Module,
                        "fitness": float,
                        "num_mazes_solved": int,
                        "avg_steps_taken": float
                    }
        """
        individual.eval()
        individual.to(device)

        total_fitness = 0.0
        maze_solved = [False for _ in train_mazes]
        steps_taken = [steps for _ in train_mazes]

        for i, maze in enumerate(train_mazes):
            visited_positions = [maze.start]
            fitness = 0.0

            start_x, start_y = maze.start
            goal_x, goal_y = maze.goal

            mouse = Mouse(start_x, start_y, Orientation.NORTH, maze)
            initial_distance = ((start_x - goal_x) ** 2 + (start_y - goal_y) ** 2) ** 0.5

            turn_count = 0
            forwards_count = 0
            moved = False

            wall_left = False
            wall_front = False
            wall_right = False

            for step in range(steps):
                wall_left, wall_front, wall_right = detect_walls(mouse, maze)
                current_distance = ((mouse.position[0] - goal_x) ** 2 + (mouse.position[1] - goal_y) ** 2) ** 0.5

                inputs = torch.tensor(
                    [
                        mouse.position[0],
                        mouse.position[1],
                        goal_x,
                        goal_y,
                        wall_left,
                        wall_front,
                        wall_right,
                        current_distance,
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    outputs = individual(inputs)
                    action = torch.argmax(outputs).item()

                if action == Action.TURN_LEFT:
                    mouse.turn_left()
                    turn_count += 1
                    forwards_count = 0

                elif action == Action.TURN_RIGHT:
                    mouse.turn_right()
                    turn_count += 1
                    forwards_count = 0

                elif action == Action.MOVE_FORWARD:
                    moved = mouse.move_forward()
                    forwards_count += 1
                    turn_count = 0

                if mouse.position not in visited_positions:
                    fitness += (initial_distance - current_distance) * 100  # Reward for moving closer to the goal
                    visited_positions.append(mouse.position)
                else:
                    fitness -= 100  # Penalty for revisiting a position

                if mouse.position == maze.goal:
                    fitness += 100000  # Reward for reaching the goal
                    maze_solved[i] = True
                    steps_taken[i] = step
                    break

                if action == 0 or action == 1 and turn_count >= 2:
                    fitness -= 100 * 1.01**turn_count  # Penalty for turning too many times

                if action == 2 and not moved and forwards_count >= 2:
                    fitness -= 100 * 1.01**forwards_count  # Penalty for moving into a wall too many time

            total_fitness += fitness / steps

        return {
            "individual": individual,
            "fitness": total_fitness / len(train_mazes),
            "num_mazes_solved": sum(maze_solved),
            "avg_steps_taken": sum(steps_taken) / len(train_mazes),
        }

    def train(self, train_mazes: list[Maze], starting_steps: int = 0, parallel_execution: bool = True) -> nn.Module:
        """Trains the population on the given mazes.

        Args:
            - train_mazes (list[Maze]): The list of mazes to train the population on.
            - starting_steps (int): The number of steps to train the population on each maze.
            - parallel_execution (bool): Whether to use parallel execution for evaluating the population.

        Returns:
            - nn.Module: The best individual after training.
        """
        best_model_age = 0
        previous_best_model = None
        steps = starting_steps
        with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as self.progress:
            self.training_progress = self.progress.add_task("[red]Training", total=self.num_generations)

            for generation in range(self.num_generations):
                if generation % 100 == 0:
                    steps += 10
                    if steps > settings.evolution.max_steps:
                        steps = settings.evolution.max_steps

                if parallel_execution:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                        results = list(
                            executor.map(
                                self.evaluate_individual,
                                self.population,
                                [train_mazes] * self.population_size,
                                [steps] * self.population_size,
                                [self.device] * self.population_size,
                            )
                        )
                else:
                    results = [
                        self.evaluate_individual(individual, train_mazes, steps, self.device)
                        for individual in self.population
                    ]

                sorted_population = sorted(results, key=lambda x: x["fitness"], reverse=True)

                best_individual = sorted_population[0]["individual"]
                best_fitness = sorted_population[0]["fitness"]
                best_mazes_solved = sorted_population[0]["num_mazes_solved"]
                best_avg_steps = sorted_population[0]["avg_steps_taken"]

                percentage_solved = (best_mazes_solved / len(train_mazes)) * 100

                self.best_model = best_individual

                if previous_best_model == self.best_model:
                    best_model_age += 1
                else:
                    best_model_age = 0

                previous_best_model = self.best_model

                best_model_mutated = MazeSolver(
                    settings.evolution.input_size, settings.evolution.hidden_size, settings.evolution.output_size
                )
                best_model_mutated.load_state_dict(self.best_model.state_dict())
                best_model_mutated = self.mutate(best_model_mutated)
                new_population = [self.best_model, best_model_mutated]

                while len(new_population) < self.population_size:
                    parent1 = random.choice(sorted_population)["individual"]
                    parent2 = random.choice(sorted_population)["individual"]

                    if torch.rand(1).item() < self.crossover_rate:
                        child1, child2 = self.crossover(parent1, parent2)
                        new_population.append(self.mutate(child1))
                        new_population.append(self.mutate(child2))

                self.population = new_population

                self.progress.console.print(
                    f"Generation {generation + 1}, "
                    f"Best Fitness: {best_fitness:.2f}, "
                    f"Reached Goal: {best_mazes_solved}/{len(train_mazes)} ({percentage_solved:.2f}%), "
                    f"Average Steps: {best_avg_steps:.2f}, "
                    f"Best Model Age: {best_model_age}"
                )

                self.progress.advance(task_id=self.training_progress)

            assert self.best_model is not None
            return self.best_model

    def save_as_state_dict(self, path: Path):
        """Saves the best model as a state dict.

        Args:
            - path (Path): The path to save the state dict.
        """
        assert self.best_model is not None
        torch.save(self.best_model.state_dict(), path)
