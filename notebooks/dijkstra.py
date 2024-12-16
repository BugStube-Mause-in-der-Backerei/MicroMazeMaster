from micromazemaster.models.maze import Maze
from micromazemaster.utils.dijkstra import dijkstra_path, plot_path

maze = Maze(width=10, height=10, seed=42)
path = dijkstra_path(maze=maze)
fig = maze.plot_graph()
plot_path(maze=maze, path=path, fig=fig)
