class Backtracker:
    count = 0

    def __init__(self):
        pass

    @classmethod
    def doBacktracking(cls, mouse):
        cls.count += 1
        maze = mouse.maze
        ways = [False, False, False]

        if mouse.position[0] == maze.goal[0] and mouse.position[1] == maze.goal[1]:
            return True

        ways[1] = maze.is_valid_move_orientation((mouse.position[0], mouse.position[1]), mouse.orientation)
        ways[0] = maze.is_valid_move_orientation((mouse.position[0], mouse.position[1]), mouse.orientation.subtract(1))
        ways[2] = maze.is_valid_move_orientation((mouse.position[0], mouse.position[1]), mouse.orientation.add(1))

        print(str(mouse.position) + " : " + str(mouse.orientation) + " : " + str(ways) + " : " + str(cls.count))

        for i in range(0, len(ways)):
            if ways[i]:
                match i:
                    case 0:
                        mouse.turn_left()
                        mouse.move_forward()
                    case 1:
                        mouse.move_forward()
                    case 2:
                        mouse.turn_right()
                        mouse.move_forward()

                result = cls.doBacktracking(mouse)

                if result == 1:
                    return 1

                match i:
                    case 0:
                        mouse.move_backward()
                        mouse.turn_right()

                    case 1:
                        mouse.move_backward()

                    case 2:
                        mouse.move_backward()
                        mouse.turn_left()
