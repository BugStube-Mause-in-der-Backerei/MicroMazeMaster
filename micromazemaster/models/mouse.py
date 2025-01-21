import math
import numpy as np
from micromazemaster.models.maze import Orientation
from micromazemaster.utils.sensors.tof import TOFSensor, SensorGroup
from micromazemaster.utils.config import settings

cell_size = settings.CELL_SIZE
offset_angle = settings.sensor.tof.OFFSET_ANGLE


class Mouse:
    def __init__(self, start_position, maze):
        self.position = [start_position[0], start_position[1]]
        self.maze = maze
        self.last_cell = [0.5, 0.5]
        self.velocity = [0.0, 0.0]
        self.acceleration = 1.0
        self.max_velocity = 2.0
        self.orientation = Orientation.NORTH
        self.target_cell = None
        self.distance = [float("inf"), float("inf"), float("inf")]
        left = SensorGroup("min")
        left.add_sensor(TOFSensor(self, 80), 0)
        left.add_sensor(TOFSensor(self, 30), 0)
        left.add_sensor(TOFSensor(self, 55), 0)

        front = SensorGroup("min")
        front.add_sensor(TOFSensor(self, 30), 0)
        front.add_sensor(TOFSensor(self, -30), 0)
        front.add_sensor(TOFSensor(self, 0), 0)
        front.add_sensor(TOFSensor(self, -10), 0)
        front.add_sensor(TOFSensor(self, 10), 0)

        right = SensorGroup("min")
        right.add_sensor(TOFSensor(self, -80), 0)
        right.add_sensor(TOFSensor(self, -30), 0)
        right.add_sensor(TOFSensor(self, -55), 0)

        self.sensors = [left, front, right]

    def set_target_cell(self, target_cell):
        self.target_cell = [round(target_cell[0], 1), round(target_cell[1], 1)]

    def move(self, dt):
        if self.target_cell:
            target_direction = [
                self.target_cell[0] - self.position[0], self.target_cell[1] - self.position[1]]
            target_distance = math.sqrt(
                target_direction[0]**2 + target_direction[1]**2)

            if target_distance > 0:
                target_direction[0] /= target_distance
                target_direction[1] /= target_distance

                self.velocity[0] = target_direction[0] * self.acceleration * dt
                self.velocity[1] = target_direction[1] * self.acceleration * dt

            pos_list = list(self.position)

            pos_list[0] += self.velocity[0]
            pos_list[1] += self.velocity[1]

            distance_to_target = math.sqrt(
                (self.position[0] - self.target_cell[0])**2 +
                (self.position[1] - self.target_cell[1])**2
            )
            if distance_to_target <= 0.2:
                self.position = self.target_cell
                self.velocity = [0.0, 0.0]
                self.target_cell = None
                return True

            self.position = tuple(pos_list)
            # self.update_sensor()
            # print(f"Position: {self.position}")
            # print(f"Orientation: {self.orientation.name}")
            # print(f"Distances: l:{round(self.distance[0], 2)}, f:{round(self.distance[1], 2)}, r:{round(self.distance[2], 2)}")

        return False

    def rotate_left(self):
        match self.orientation:
            case Orientation.NORTH:
                self.orientation = Orientation.WEST
            case Orientation.EAST:
                self.orientation = Orientation.NORTH
            case Orientation.SOUTH:
                self.orientation = Orientation.EAST
            case Orientation.WEST:
                self.orientation = Orientation.SOUTH
        self.velocity = [0.0, 0.0]

    def rotate_right(self):
        match self.orientation:
            case Orientation.NORTH:
                self.orientation = Orientation.EAST
            case Orientation.EAST:
                self.orientation = Orientation.SOUTH
            case Orientation.SOUTH:
                self.orientation = Orientation.WEST
            case Orientation.WEST:
                self.orientation = Orientation.NORTH
        self.velocity = [0.0, 0.0]

    def update_sensor(self):
        for i in range(len(self.sensors)):
            new_distance, _ = self.sensors[i].get_distance(self.maze)
            if new_distance is None:
                self.distance[i] = float("inf")
            else:
                self.distance[i] = new_distance
                # print(f"{self.orientation.name}\n{new_distance}")
