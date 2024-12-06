from micromazemaster.models.maze import Maze
from micromazemaster.models.mouse import Mouse, Orientation
import matplotlib.pyplot as plt
from micromazemaster.utils.config import settings

cell_size = settings.CELL_SIZE

maze = Maze(width=5, height=5, seed=1234, missing_walls=False)
mouse = Mouse(x=0.5, y=0.5, orientation=Orientation.EAST, maze=maze)

# The update_sensor method will update the mouse's distance attribute
mouse.update_sensor()
print(mouse.distance)

# The move method will move the mouse
print(mouse.position)
mouse.move_forward()
print(mouse.position)

# The turn method will turn the mouse
print(mouse.orientation)
mouse.turn_left()
print(mouse.orientation)

# Plot the maze and the sensors
fig, ax = maze.plot()

for sensor in [mouse.TOFSensorLeft, mouse.TOFSensorCenter, mouse.TOFSensorRight]:
    distance, intersection_point = sensor.get_distance(maze)
    distance, intersection_point = distance / cell_size, (intersection_point[0] / cell_size, intersection_point[1] / cell_size)
    ax.plot([sensor.x / cell_size, intersection_point[0]], [sensor.y / cell_size, intersection_point[1]], 'r--', label='Ray')
    print(f"Sensor at ({sensor.x}, {sensor.y}) with angle {sensor.angle}: Distance to the wall = {distance} mm")

ax.plot(sensor.x / cell_size, sensor.y / cell_size, 'bo', label='Sensor', markersize=20)

plt.xticks(range(0, 11))
plt.yticks(range(0, 11))
plt.grid()
plt.axis("on")
plt.show()