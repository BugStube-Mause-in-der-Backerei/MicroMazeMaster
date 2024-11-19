from micromazemaster.models.maze import Orientation


class Mouse:
    def __init__(self, x, y, orientation, maze):
        self.position = (x, y)
        self.orientation = orientation
        self.maze = maze

    def move_forward(self):
        if self.maze.is_valid_move(self.position, self.orientation):
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
