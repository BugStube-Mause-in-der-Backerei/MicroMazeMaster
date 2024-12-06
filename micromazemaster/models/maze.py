import json
import random
from collections import deque
from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString

from micromazemaster.utils.config import settings
from micromazemaster.utils.logging import logger


class Orientation(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Cell:
    def __init__(self):
        self.n = True
        self.s = True
        self.e = True
        self.w = True


class Wall:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_position = (start_x, start_y)
        self.end_position = (end_x, end_y)

    def to_dict(self):
        return {"start_position": self.start_position, "end_position": self.end_position}

    @classmethod
    def from_dict(cls, wall_dict):
        start_x, start_y = wall_dict["start_position"]
        end_x, end_y = wall_dict["end_position"]
        return cls(start_x, start_y, end_x, end_y)

    def get_positions(self):
        return self.start_position, self.end_position


class Maze:
    def __init__(self, width, height, seed, missing_walls=settings.WALLS_TO_REMOVE, generation=True):
        self.seed = seed
        self.width = width
        self.height = height
        self.missing_walls = missing_walls
        self.walls = []
        self.shapely_walls = []
        self.graph = nx.Graph()
        self.goal = ()
        self.start = (0.5, 0.5)
        if generation:
            self.__generate_maze()
        self.map = self.__generate_map()

    def to_dict(self):
        return {
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "missing_walls": self.missing_walls,
            "walls": [wall.to_dict() for wall in self.walls],
        }

    def __generate_maze(self):
        random.seed(self.seed)

        while True:
            # Generate half-coordinate positions (0.5, 1.5, ..., width-0.5)
            x = random.randint(0, self.width - 1) + 0.5
            y = random.randint(0, self.height - 1) + 0.5

            # Ensure the goal is not the start position
            if (x, y) != self.start:
                self.goal = (x, y)
                break

        # Initialize map and visited list
        map_cells = [Cell() for _ in range(self.width * self.height)]
        visited = [False] * (self.width * self.height)

        # Stack for backtracking
        stack = deque()
        current = 0
        stack.append(current)

        while stack:
            visited[current] = True
            options = []

            # Check possible directions (up, down, left, right)
            if current >= self.width and not visited[current - self.width]:  # up
                options.append(current - self.width)
            if current < self.width * self.height - self.width and not visited[current + self.width]:  # down
                options.append(current + self.width)
            if current % self.width != 0 and not visited[current - 1]:  # left
                options.append(current - 1)
            if (current + 1) % self.width != 0 and not visited[current + 1]:  # right
                options.append(current + 1)

            if not options:
                current = stack.pop()  # Backtrack if no unvisited neighbors
            else:
                random.shuffle(options)
                new_cell = options[0]

                # Remove the wall between current and new_cell
                diff = new_cell - current
                if diff == -1:  # left
                    map_cells[current].w = False
                    map_cells[new_cell].e = False
                elif diff == 1:  # right
                    map_cells[current].e = False
                    map_cells[new_cell].w = False
                elif diff == -self.width:  # up
                    map_cells[current].n = False
                    map_cells[new_cell].s = False
                elif diff == self.width:  # down
                    map_cells[current].s = False
                    map_cells[new_cell].n = False
                else:
                    raise ValueError("Unexpected case while removing wall.")

                stack.append(current)
                current = new_cell

        # Now, generate the walls from the cell information
        start_x, start_y = 0, 0

        # Horizontal walls (south walls)
        for y in range(self.height - 1):
            in_wall = False
            for x in range(self.width):
                if not in_wall and map_cells[y * self.width + x].s:
                    start_x = x
                    in_wall = True
                elif in_wall and not map_cells[y * self.width + x].s:
                    self.walls.append(Wall(start_x, y + 1, x, y + 1))
                    in_wall = False
            if in_wall:
                self.walls.append(Wall(start_x, y + 1, self.width, y + 1))

        # Vertical walls (east walls)
        for x in range(self.width - 1):
            in_wall = False
            for y in range(self.height):
                if not in_wall and map_cells[y * self.width + x].e:
                    start_y = y
                    in_wall = True
                elif in_wall and not map_cells[y * self.width + x].e:
                    self.walls.append(Wall(x + 1, start_y, x + 1, y))
                    in_wall = False
            if in_wall:
                self.walls.append(Wall(x + 1, start_y, x + 1, self.height))

        # Remove some walls
        # Check if the number of walls is bigger than the number of walls to remove
        if len(self.walls) > self.missing_walls:
            indices_to_remove = random.sample(range(len(self.walls)), self.missing_walls)
            for wall in sorted(indices_to_remove, reverse=True):
                self.walls.pop(wall)

        # Outer boundary walls
        self.walls.append(Wall(0, 0, self.width, 0))  # top
        self.walls.append(Wall(self.width, 0, self.width, self.height))  # right
        self.walls.append(Wall(self.width, self.height, 0, self.height))  # bottom
        self.walls.append(Wall(0, self.height, 0, 0))

        # Convert walls to shapely objects
        self.shapely_walls = [LineString(wall.get_positions()) for wall in self.walls]
        self.__generate_graph()

    def __generate_graph(self):
        for x in range(self.width):
            for y in range(self.height):
                self.graph.add_node((x + 0.5, y + 0.5))

        for x in range(self.width):
            for y in range(self.height):
                current_node = (x + 0.5, y + 0.5)
                neighbors = [
                    (x + 0.5, y + 1.5),  # North
                    (x + 1.5, y + 0.5),  # East
                    (x + 0.5, y - 0.5),  # South
                    (x - 0.5, y + 0.5),  # West
                ]
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                        line = LineString([current_node, neighbor])
                        if not any(line.intersects(wall) for wall in self.shapely_walls):
                            self.graph.add_edge(current_node, neighbor, weight=1)

    def __generate_image(self, cell_size=20):
        image = Image.new("1", (self.width * cell_size + 1, self.height * cell_size + 1))

        draw = ImageDraw.Draw(image)

        for wall in self.walls:
            line = (
                (wall.start_position[0] * cell_size, image.height - wall.start_position[1] * cell_size - 1),
                (wall.end_position[0] * cell_size, image.height - wall.end_position[1] * cell_size - 1),
            )
            draw.line(line, fill=1)

        return image

    def show(self, cell_size=20):
        image = self.__generate_image(cell_size)
        image.show()

    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(self.width, self.height))
        ax = fig.add_subplot(111)
        for wall in self.shapely_walls:
            ax.plot(*wall.xy, color="black", linewidth=5)
        plt.axis("equal")
        plt.axis("off")
        return fig, ax

    def export_as_png(self, path, cell_size=20):
        image = self.__generate_image(cell_size)
        image.save(path, "PNG")

    def export_as_json(self, path):
        try:
            with open(path, "w") as file:
                json.dump(self.to_dict(), file)
        except Exception as e:
            logger.error(f"Error writing to file: {e}")

    def is_valid_move(self, position, orientation):

        match orientation:
            case Orientation.NORTH:
                new_position = (position[0], position[1] + 1)
            case Orientation.EAST:
                new_position = (position[0] + 1, position[1])
            case Orientation.SOUTH:
                new_position = (position[0], position[1] - 1)
            case Orientation.WEST:
                new_position = (position[0] - 1, position[1])

        return new_position in nx.neighbors(self.graph, position)

    def is_valid_move(self, position, new_position):
        return new_position in nx.neighbors(self.graph, position)

    @classmethod
    def from_json(cls, path):
        try:
            with open(path, "r") as file:
                data = json.load(file)
                maze = cls(data["width"], data["height"], data["seed"], data["missing_walls"], generation=False)
                maze.walls = [Wall.from_dict(wall_data) for wall_data in data["walls"]]
                return maze
        except Exception as e:
            logger.error(f"Error reading from file: {e}")
            return None

    def __generate_map(self, cell_size=settings.CELL_SIZE-1):
        arr = np.array(self.__generate_image(cell_size), dtype=np.uint8)
        return arr[::-1]
