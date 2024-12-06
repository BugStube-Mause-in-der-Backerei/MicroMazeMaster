import math
import numpy as np
from micromazemaster.models.maze import Maze
from typing import Optional, Tuple
from micromazemaster.utils.config import settings

step_size = settings.STEP_SIZE
max_distance = settings.MAX_DISTANCE

class TOFSensor:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def get_distance(self, maze: Maze) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """Returns the distance (in mm) to the nearest wall in the direction of the sensor.

        Args:
            - maze (Maze): The maze object that contains the walls.

        Returns:
            - Tuple[float | None, Point | None]: The distance to the nearest wall in mm and the intersection point.
        """

        ray_dx = math.cos(math.radians(self.angle))
        ray_dy = math.sin(math.radians(self.angle))

        for distance in np.arange(0, max_distance, step_size):
            ray_x = self.x + distance * ray_dx
            ray_y = self.y + distance * ray_dy

            grid_x = int(ray_x)
            grid_y = int(ray_y)

            if maze.map[grid_y, grid_x] == 1:
                distance = math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)
                return distance, (ray_x, ray_y)

        return None, None