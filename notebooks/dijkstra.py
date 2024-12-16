# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
import matplotlib.pyplot as plt

from micromazemaster.models.maze import Maze
from micromazemaster.utils.dijkstra import dijkstra_path, plot_path

# +
maze = Maze(width=10, height=10, seed=42)
path = dijkstra_path(maze=maze)
fig = maze.plot_graph()
plot_path(maze=maze, path=path, fig=fig)
plt.show()
