import math
from typing import Optional, Tuple

import numpy as np

from micromazemaster.models.maze import Maze
from micromazemaster.utils.config import settings

step_size = settings.sensor.tof.step_size
max_distance = settings.sensor.tof.max_distance
cell_size = settings.cell_size

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
            ray_x = self.mouse.position[0] * cell_size + distance * ray_dx
            ray_y = self.mouse.position[1] * cell_size + distance * ray_dy

            grid_x = int(ray_x)
            grid_y = int(ray_y)

            if maze.map[grid_y, grid_x] == 1:
                distance = math.sqrt(
                    (ray_x - self.mouse.position[0] * cell_size) ** 2
                    + (ray_y - self.mouse.position[1] * cell_size) ** 2
                )
                return distance, (ray_x, ray_y)

        return None, None


class SensorGroup:
    def __init__(self, mode):
        self.sensors = []
        self.mode = mode
        self.noise_func = None

    def add_sensor(self, sensor: TOFSensor, offset_value: float):
        self.sensors.append((sensor, offset_value))

    def get_distance(self, maze: Maze) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        data = []

        for sensor in self.sensors:
            raw, point = sensor[0].get_distance(maze)
            raw += sensor[1]
            if self.noise_func:
                raw = self.noise_func(raw)
            data.append((raw, point))

        match self.mode:
            case "min":
                min_tuple = min(data, key=lambda x: x[0])
                return min_tuple
            case "max":
                max_tuple = max(data, key=lambda x: x[0])
                return max_tuple
            case "mean":
                mean_value = np.mean([x[0] for x in data])
                closest_tuple = min(data, key=lambda x: abs(x[0] - mean_value))
                return closest_tuple
            case "median":
                median_value = np.median([x[0] for x in data])
                closest_tuple = min(data, key=lambda x: abs(x[0] - median_value))
                return closest_tuple
            case _:
                min_tuple = min(data, key=lambda x: x[0])
                return min_tuple