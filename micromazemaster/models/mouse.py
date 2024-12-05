from micromazemaster.models.maze import Orientation
from micromazemaster.utils.config import settings
from micromazemaster.utils.sensors.tof import TOFSensor

cell_size = settings.CELL_SIZE
offset_angel = settings.OFFSET_ANGLE


class Mouse:
    def __init__(self, x, y, orientation, maze):
        self.position = (x, y)
        self.orientation = orientation
        self.maze = maze
        self.distance = [float("inf"), float("inf"), float("inf")]
        # TOF sensor needs coordinates in units
        self.TOFSensorLeft = TOFSensor(
            self.position[0] * cell_size, self.position[1] * cell_size, (self.orientation.value - 1) * 90 + offset_angel
        )
        self.TOFSensorCenter = TOFSensor(
            self.position[0] * cell_size, self.position[1] * cell_size, (self.orientation.value - 1) * 90
        )
        self.TOFSensorRight = TOFSensor(
            self.position[0] * cell_size, self.position[1] * cell_size, (self.orientation.value - 1) * 90 - offset_angel
        )

    def move_forward(self):
        if self.maze.is_valid_move_orientation(self.position, self.orientation):
            match self.orientation:
                case Orientation.NORTH:
                    self.position = (self.position[0], self.position[1] + 1)
                case Orientation.EAST:
                    self.position = (self.position[0] + 1, self.position[1])
                case Orientation.SOUTH:
                    self.position = (self.position[0], self.position[1] - 1)
                case Orientation.WEST:
                    self.position = (self.position[0] - 1, self.position[1])
            return True
        else:
            return False

    def turn_left(self):
        match self.orientation:
            case Orientation.NORTH:
                self.orientation = Orientation.WEST
            case Orientation.EAST:
                self.orientation = Orientation.NORTH
            case Orientation.SOUTH:
                self.orientation = Orientation.EAST
            case Orientation.WEST:
                self.orientation = Orientation.SOUTH

    def turn_right(self):
        match self.orientation:
            case Orientation.NORTH:
                self.orientation = Orientation.EAST
            case Orientation.EAST:
                self.orientation = Orientation.SOUTH
            case Orientation.SOUTH:
                self.orientation = Orientation.WEST
            case Orientation.WEST:
                self.orientation = Orientation.NORTH

    def move_backward(self):
        if self.maze.is_valid_move_orientation(self.position, self.orientation.subtract(2)):
            match self.orientation:
                case Orientation.NORTH:
                    self.position = (self.position[0], self.position[1] - 1)
                case Orientation.EAST:
                    self.position = (self.position[0] - 1, self.position[1])
                case Orientation.SOUTH:
                    self.position = (self.position[0], self.position[1] + 1)
                case Orientation.WEST:
                    self.position = (self.position[0] + 1, self.position[1])
            return True
        else:
            return False

    def update_sensor(self):
        self.distance[0], _ = self.TOFSensorLeft.get_distance(self.maze)
        self.distance[1], _ = self.TOFSensorCenter.get_distance(self.maze)
        self.distance[2], _ = self.TOFSensorRight.get_distance(self.maze)

    def move_backward(self):
        if self.maze.is_valid_move_orientation(self.position, self.orientation.subtract(2)):
            match self.orientation:
                case Orientation.NORTH:
                    self.position = (self.position[0], self.position[1] - 1)
                case Orientation.EAST:
                    self.position = (self.position[0] - 1, self.position[1])
                case Orientation.SOUTH:
                    self.position = (self.position[0], self.position[1] + 1)
                case Orientation.WEST:
                    self.position = (self.position[0] + 1, self.position[1])
            return True
        else:
            return False
