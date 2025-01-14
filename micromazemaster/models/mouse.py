import math
import numpy as np
from micromazemaster.models.maze import Orientation
from micromazemaster.utils.sensors.tof import TOFSensor
from micromazemaster.utils.config import settings

cell_size = settings.CELL_SIZE
offset_angel = settings.OFFSET_ANGLE
class Mouse:
    def __init__(self, start_position, maze):
        self.position = [start_position[0], start_position[1]]
        self.maze = maze
        self.velocity = [0.0, 0.0]
        self.acceleration = 1.0
        self.max_velocity = 2.0
        self.orientation = 0.0
        self.target_cell = None
        self.distance = [float("inf"), float("inf"), float("inf")]
        # TOF sensor needs coordinates in units
        self.TOFSensorLeft = TOFSensor(self.position[0]*cell_size, self.position[1]*cell_size, (self.orientation % (2 * math.pi)) + offset_angel)
        self.TOFSensorCenter = TOFSensor(self.position[0]*cell_size, self.position[1]*cell_size, (self.orientation % (2 * math.pi)))
        self.TOFSensorRight = TOFSensor(self.position[0]*cell_size, self.position[1]*cell_size, (self.orientation % (2 * math.pi)) - offset_angel)


    def set_target_cell(self, target_cell):
        """Set the cell the mouse is moving toward."""
        self.target_cell = [round(target_cell[0], 1), round(target_cell[1], 1)]

    def move(self, dt):
        # Calculate the direction vector toward the target cell
        if self.target_cell:
            target_direction = [self.target_cell[0] - self.position[0], self.target_cell[1] - self.position[1]]
            target_distance = math.sqrt(target_direction[0]**2 + target_direction[1]**2)

            # Normalize the direction vector if the distance is non-zero
            if target_distance > 0:
                target_direction[0] /= target_distance
                target_direction[1] /= target_distance

                # Apply acceleration in the direction of the target
                self.velocity[0] = target_direction[0] * self.acceleration * dt
                self.velocity[1] = target_direction[1] * self.acceleration * dt

                # Normalize the velocity if it exceeds max_velocity
                velocity_magnitude = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
                if velocity_magnitude > self.max_velocity:
                    scale = self.max_velocity / velocity_magnitude
                    self.velocity[0] *= scale
                    self.velocity[1] *= scale

            pos_list = list(self.position)

            # Update position based on velocity
            pos_list[0] += self.velocity[0]
            pos_list[1] += self.velocity[1]

            self.position = tuple(pos_list)

            # Check if the center of the target cell is reached
            distance_to_target = math.sqrt(
                (self.position[0] - self.target_cell[0])**2 +
                (self.position[1] - self.target_cell[1])**2
            )
            if distance_to_target < 0.1:
                self.position = self.target_cell
                self.velocity = [0.0, 0.0]
                self.target_cell = None
                return True
        return False


    def rotate(self, angle):
        self.orientation = (self.orientation + angle) % (2 * math.pi)
        self.velocity = [0.0, 0.0]

    def update_sensor(self):
        self.distance[0], _ = self.TOFSensorLeft.get_distance(self.maze)
        self.distance[1], _ = self.TOFSensorCenter.get_distance(self.maze)
        self.distance[2], _ = self.TOFSensorRight.get_distance(self.maze)
