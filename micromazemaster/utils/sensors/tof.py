import math
import numpy as np
from micromazemaster.models.maze import Maze
from typing import Optional, Tuple
from micromazemaster.utils.config import settings

step_size = settings.STEP_SIZE
max_distance = settings.MAX_DISTANCE
cell_size = settings.CELL_SIZE

class TOFSensor:
    def __init__(self, mouse, offset_angle=0):
        self.mouse = mouse
        self.offset_angle = offset_angle
        self.sensor_angle = self.get_angle()

    def get_angle(self):
        angle = (90 - self.mouse.orientation.value * 90) + self.offset_angle
        if angle < 0:
            angle += 360
        return angle

    def get_distance(self, maze: Maze) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """Returns the distance (in mm) to the nearest wall in the direction of the sensor.

        Args:
            - maze (Maze): The maze object that contains the walls.

        Returns:
            - Tuple[float | None, Point | None]: The distance to the nearest wall in mm and the intersection point.
        """

        ray_dx = math.cos(math.radians(self.get_angle()))
        ray_dy = math.sin(math.radians(self.get_angle()))

        for distance in np.arange(0, max_distance, step_size):
            ray_x = self.mouse.position[0]*cell_size + distance * ray_dx
            ray_y = self.mouse.position[1]*cell_size + distance * ray_dy

            grid_x = int(ray_x)
            grid_y = int(ray_y)

            if maze.map[grid_y, grid_x] == 1:
                distance = math.sqrt((ray_x - self.mouse.position[0]*cell_size) ** 2 + (ray_y - self.mouse.position[1]*cell_size) ** 2)
                return distance, (ray_x, ray_y)

        return None, None