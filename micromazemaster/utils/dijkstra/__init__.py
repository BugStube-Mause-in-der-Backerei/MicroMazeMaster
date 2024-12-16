import matplotlib.pyplot as plt
import networkx as nx

from micromazemaster.models.maze import Maze


def dijkstra_path(maze: Maze) -> list[tuple[int, int]]:
    """Returns the path from start to end using Dijkstra's algorithm.

    Args:
        - maze (Maze): The maze object that contains the walls.

    Returns:
        - list[tuple[int, int]]: The path from start to end.
    """
    return nx.dijkstra_path(maze.graph, maze.start, maze.goal)


def plot_path(maze: Maze, fig: plt.figure, path: list[tuple[int, int]]) -> None:
    """Plots the path on the maze.

    Args:
        - fig (plt.figure): The figure object to plot on.
        - maze (Maze): The maze object that contains the graph.
        - path (list[tuple[int, int]]): The path from start to end.
    """
    graph = maze.graph
    plt.figure(fig.number)

    pos = {node: (node[0], node[1]) for node in graph.nodes()}
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="blue", width=2)
    nx.draw_networkx_nodes(graph, pos, nodelist=[path[0]], node_color="green", node_size=300, label="Start")
    nx.draw_networkx_nodes(graph, pos, nodelist=[path[-1]], node_color="red", node_size=300, label="Goal")
    plt.title("Shortest Path in Maze using Dijkstra's Algorithm")
