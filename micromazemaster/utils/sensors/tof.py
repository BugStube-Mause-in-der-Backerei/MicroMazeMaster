import math

from shapely.geometry import LineString, Point

from micromazemaster.models.maze import Maze
from micromazemaster.utils.config import settings


class TOFSensor:
    def __init__(self, x, y, angle, max_distance):
        self.x = x
        self.y = y
        self.angle = angle
        self.max_distance = max_distance

    def get_distance(self, maze: Maze) -> tuple[float | None, Point | None]:
        """Returns the distance (in mm) to the nearest wall in the direction of the sensor.

        Args:
            - maze (Maze): The maze object that contains the walls.

        Returns:
            - Tuple[float | None, Point | None]: The distance to the nearest wall in mm and the intersection point.
        """

        sensor_pos = Point(self.x, self.y)
        ray_end = Point(
            self.x + self.max_distance * math.cos(math.radians(self.angle)),
            self.y + self.max_distance * math.sin(math.radians(self.angle)),
        )

        ray = LineString([sensor_pos, ray_end])

        min_distance = float("inf")
        min_distance_point = None

        for wall in maze.shapely_walls:
            if ray.intersects(wall):
                intersection = ray.intersection(wall)
                if isinstance(intersection, Point):
                    distance = sensor_pos.distance(intersection)
                    min_distance = min(min_distance, distance)
                    if distance == min_distance:
                        min_distance_point = intersection

        if min_distance == float("inf"):
            return None, None

        min_distance *= settings.GRID_TO_MM

        return min_distance, min_distance_point
