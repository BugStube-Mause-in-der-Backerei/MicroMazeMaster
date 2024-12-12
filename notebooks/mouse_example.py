from micromazemaster.models.maze import Maze
from micromazemaster.models.mouse import Mouse, Orientation
import matplotlib.pyplot as plt
from micromazemaster.utils.config import settings

cell_size = settings.CELL_SIZE

maze = Maze(width=5, height=5, seed=1234, missing_walls=False)
mouse = Mouse(x=0.5, y=0.5, orientation=Orientation.EAST, maze=maze)

# The update_sensor method will update the mouse's distance attribute

mouse.update_sensor()
print("Sensor distances" + str(mouse.distance))

# The move method will move the mouse
print("Position before move" + str(mouse.position))
mouse.move_forward()
print("Position after move" + str(mouse.position))

# The turn method will turn the mouse
# print("Orientation before move" + str(mouse.orientation))
# print("Angle before move" + str(mouse.sensors[1].get_angle()))
# mouse.turn_right()
# print("Orientation after move" + str(mouse.orientation))
# print("Angle after move" + str(mouse.sensors[1].get_angle()))

# Plot the maze and the sensors
fig, ax = maze.plot()

for sensor in mouse.sensors:
    distance, intersection_point = sensor.get_distance(maze)
    distance, intersection_point = distance / cell_size, (intersection_point[0] / cell_size, intersection_point[1] / cell_size)
    ax.plot([sensor.mouse.position[0], intersection_point[0]], [sensor.mouse.position[1], intersection_point[1]], 'r--', label='Ray')
    print(f"Sensor at ({sensor.mouse.position[0]}, {sensor.mouse.position[1]}) with angle {sensor.get_angle()}: Distance to the wall = {distance} mm")

ax.plot(mouse.sensors[1].mouse.position[0], mouse.sensors[1].mouse.position[1], 'bo', label='Sensor', markersize=20)

plt.xticks(range(0, 6))
plt.yticks(range(0, 6))
plt.grid()
plt.axis("on")
plt.show()
plt.show()