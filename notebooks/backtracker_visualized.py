from collections import Counter

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from micromazemaster.models.maze import Maze
from micromazemaster.models.mouse import Mouse, Orientation

STARTING_POSITION = (0.5, 0.5)
GOAL_POSITION = (9.5, 4.5)


class MazeEnv:
    def __init__(self, maze, starting_position=STARTING_POSITION, goal_position=GOAL_POSITION):
        self.maze = maze
        self.width = maze.width
        self.height = maze.height
        self.start_position = starting_position
        self.goal = goal_position
        self.walls = maze.walls


class Backtracker:
    count = 0

    def __init__(self):
        pass

    @classmethod
    def doBacktracking(cls, mouse, path):
        cls.count += 1
        maze = mouse.maze
        ways = [False, False, False]

        # Add the current position to the path
        path.append(tuple(mouse.position))

        if mouse.position[0] == maze.goal[0] and mouse.position[1] == maze.goal[1]:
            return True

        ways[1] = maze.is_valid_move_orientation((mouse.position[0], mouse.position[1]), mouse.orientation)
        ways[0] = maze.is_valid_move_orientation((mouse.position[0], mouse.position[1]), mouse.orientation.subtract(1))
        ways[2] = maze.is_valid_move_orientation((mouse.position[0], mouse.position[1]), mouse.orientation.add(1))

        print(str(mouse.position) + " : " + str(mouse.orientation) + " : " + str(ways) + " : " + str(cls.count))

        for i in reversed(range(len(ways))):
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

                result = cls.doBacktracking(mouse, path)

                if result:
                    return True

                # Backtrack to the previous state
                match i:
                    case 0:
                        mouse.move_backward()
                        mouse.turn_right()

                    case 1:
                        mouse.move_backward()

                    case 2:
                        mouse.move_backward()
                        mouse.turn_left()

                path.append(tuple(mouse.position))

        # Remove the position from the path if it's a dead end
        # path.pop()
        return False


def visualize_agent_run(env, positions, total_reward, total_steps, caption="Backtracking Visualization"):
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    cm_to_units = 1 / 2.54  # Conversion from cm to inches
    margin = 1 * cm_to_units  # 1cm margin

    ax.set_xlim(-margin, env.width + margin)
    ax.set_ylim(-margin, env.height + margin)
    ax.set_aspect("equal", adjustable="box")

    # Draw static elements (walls, start, goal)
    for wall in env.walls:
        (x1, y1), (x2, y2) = wall.start_position, wall.end_position
        ax.plot([x1, x2], [y1, y2], "k", linewidth=3)  # Thicker walls

    # Draw the path start and goal
    ax.plot(env.start_position[0], env.start_position[1], "go", markersize=10)
    ax.plot(env.goal[0], env.goal[1], "ro", markersize=10)

    # Draw the heatmap-style path
    position_counts = Counter(positions)
    max_visits = max(position_counts.values())
    colors = ["lightblue", "blue"]
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", colors)
    norm = mcolors.Normalize(vmin=1, vmax=max_visits)

    for i in range(len(positions) - 1):
        x_values = [positions[i][0], positions[i + 1][0]]
        y_values = [positions[i][1], positions[i + 1][1]]
        color = cmap(norm(position_counts[positions[i + 1]]))
        ax.plot(x_values, y_values, color=color, linewidth=2)

    # Initialize the agent's position as a blue dot
    (agent_dot,) = ax.plot([], [], "bo", markersize=8)

    # Animation update function
    def update(frame):
        agent_dot.set_data([positions[frame][0]], [positions[frame][1]])
        return (agent_dot,)

    # Create animation
    _ = animation.FuncAnimation(fig, update, frames=len(positions), interval=100, blit=True)

    # Remove gridlines and axis labels
    ax.axis("off")

    # Add performance stats to the side legend
    side_legend_text = f"Steps: {total_steps}\nReward: {int(total_reward)}"
    fig.text(
        0.05,
        0.79,
        side_legend_text,
        fontsize=10,
        va="center",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8),
        transform=fig.transFigure,
    )

    # Add caption above the plot, if provided
    if caption:
        fig.suptitle(caption, fontsize=12, weight="bold", y=0.95)

    # Show the animation
    plt.show()


# Example usage
maze = Maze(width=10, height=5, seed=42)
maze.goal = GOAL_POSITION
mouse = Mouse(x=maze.start[0], y=maze.start[1], orientation=Orientation.EAST, maze=maze)
env = MazeEnv(maze)
backtracker = Backtracker()
path = []
backtracker.doBacktracking(mouse, path)

visualize_agent_run(env, path, total_reward=0, total_steps=len(path), caption="Backtracking Visualization")
