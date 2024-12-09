from micromazemaster.models.maze import Maze
import random
from collections import Counter
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import animation

# ============ PARAMETERS ============ #
MAZE_SIZE = (10, 5)
SEED = 20
ACTION_SEED = 5
NUM_MAZES = 20
NUM_TEST_RUNS = 100
STARTING_POSITION = (0.5, 0.5)
GOAL_POSITION = (9.5, 4.5)
MAX_STEPS_PER_EPISODE = 300

# ============ ENVIRONMENT LOADING ============ #

action_random = random.Random(ACTION_SEED)

def generate_mazes(seed):
    random.seed(seed)
    mazes = []
    for index in range(NUM_MAZES):
        mazes.append(Maze(MAZE_SIZE[0], MAZE_SIZE[1], seed=random.randint(1, 1000)))
    random.seed(None)
    return mazes

# ============ ENVIRONMENT CLASS ============ #
class MazeEnv:
    def __init__(self, maze, starting_position=STARTING_POSITION, goal_position=GOAL_POSITION):
        self.maze = maze
        self.width = maze.width
        self.height = maze.height
        self.start_position = starting_position
        self.goal_position = goal_position
        self.walls = maze.walls
        self.reset()

    def reset(self):
        self.position = self.start_position
        self.done = False
        return self.position

    def hits_wall(self, position, new_position):
        return not self.maze.is_valid_move_position(position, new_position)
    
    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        action_names = ["North", "South", "West", "East"]
        dx, dy = moves[action]
        new_position = (self.position[0] + dx, self.position[1] + dy)

        if not self.hits_wall(self.position, new_position):
            self.position = new_position
        if self.position == self.goal_position:
            self.done = True

        return self.position, self.done, action_names[action]

# ============ VISUALIZATION ============ #
def visualize_agent_run(env, positions, total_steps, caption="Zufällige Entscheidungen auf ungesehenes Labyrinth"):
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    cm_to_units = 1 / 2.54  # Conversion from cm to inches
    margin = 1 * cm_to_units  # 1cm margin

    ax.set_xlim(-margin, env.width + margin)
    ax.set_ylim(-margin, env.height + margin)
    ax.set_aspect('equal', adjustable='box')

    # Draw static elements (walls, start, goal)
    for wall in env.walls:
        (x1, y1), (x2, y2) = wall.start_position, wall.end_position
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=3)  # Thicker walls

    # Draw the path start and goal
    ax.plot(env.start_position[0], env.start_position[1], 'go', markersize=10)
    ax.plot(env.goal_position[0], env.goal_position[1], 'ro', markersize=10)

    # Draw the heatmap-style path
    position_counts = Counter(positions)
    max_visits = max(position_counts.values())
    colors = ["lightblue", "blue", "darkblue"]
    cmap = mcolors.LinearSegmentedColormap.from_list("gradient", colors)
    norm = mcolors.Normalize(vmin=1, vmax=max_visits)

    for i in range(len(positions) - 1):
        x_values = [positions[i][0], positions[i + 1][0]]
        y_values = [positions[i][1], positions[i + 1][1]]
        color = cmap(norm(position_counts[positions[i + 1]]))
        ax.plot(x_values, y_values, color=color, linewidth=2)

    # Initialize the agent's position as a blue dot
    agent_dot, = ax.plot([], [], 'bo', markersize=8)

    # Animation update function
    def update(frame):
        agent_dot.set_data([positions[frame][0]], [positions[frame][1]])
        return agent_dot,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=10, blit=True)

    # Remove gridlines and axis labels
    ax.axis('off')

    # Add performance stats to the side legend
    side_legend_text = f"Steps: {total_steps}"
    fig.text(0.05, 0.79, side_legend_text, fontsize=10, va='center', ha='left', 
         bbox=dict(facecolor='white', alpha=0.8), transform=fig.transFigure)

    # Add caption above the plot, if provided
    if caption:
        fig.suptitle(caption, fontsize=12, weight='bold', y=0.95)

    # Show the animation
    plt.show()

def visualize_multiple_runs(env, all_positions, num_colors=20, caption="Zusammengefasste zufällige Entscheidungen auf ungesehenes Labyrinth"):
   
    # Flatten all positions into a single list
    aggregated_positions = [pos for positions in all_positions for pos in positions]
    position_counts = Counter(aggregated_positions)
    max_visits = max(position_counts.values())
    
    # Adjust the colormap to have 'num_colors' steps
    colors = plt.cm.Blues(np.linspace(0.1, 1, num_colors))  # Gradation from light to dark blue
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        boundaries=np.linspace(1, max_visits + 1, num_colors + 1),  # Define boundaries dynamically
        ncolors=num_colors
    )

    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    cm_to_units = 1 / 2.54  # Conversion from cm to inches
    margin = 1 * cm_to_units  # 1cm margin

    ax.set_xlim(-margin, env.width + margin)
    ax.set_ylim(-margin, env.height + margin)
    ax.set_aspect('equal', adjustable='box')

    # Draw static elements (walls, start, goal)
    for wall in env.walls:
        (x1, y1), (x2, y2) = wall.start_position, wall.end_position
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=3)  # Thicker walls

    # Draw the path start and goal
    ax.plot(env.start_position[0], env.start_position[1], 'go', markersize=10)
    ax.plot(env.goal_position[0], env.goal_position[1], 'ro', markersize=10)

    # Draw the heatmap-style paths
    for positions in all_positions:
        for i in range(len(positions) - 1):
            x_values = [positions[i][0], positions[i + 1][0]]
            y_values = [positions[i][1], positions[i + 1][1]]
            color = cmap(norm(position_counts[positions[i + 1]]))
            ax.plot(x_values, y_values, color=color, linewidth=2)

    # Remove gridlines and axis labels
    ax.axis('off')

    # Add a legend at the top-left corner with aggregated statistics
    total_runs = len(all_positions)
    goals_reached = sum(1 for positions in all_positions if positions[-1] == env.goal_position)
    goals_missed = total_runs - goals_reached
    avg_steps = np.mean([len(positions) for positions in all_positions])

    legend_text = (
        f"Total Runs: {total_runs}\n"
        f"Goals Reached: {goals_reached}\n"
        f"Goals Missed: {goals_missed}\n"
        f"Average Steps: {avg_steps:.0f}"
    )

   # Add performance stats to the side legend
    fig.text(0.02, 0.76, legend_text, fontsize=10, va='center', ha='left', 
         bbox=dict(facecolor='white', alpha=0.8), transform=fig.transFigure)

    # Add caption above the plot, if provided
    if caption:
        fig.suptitle(caption, fontsize=12, weight='bold', y=0.95)

    # Show the plot
    plt.show()

# ============ RANDOM AGENT CLASS ============ #
class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self):
        return action_random.choice(range(self.action_size))

# ============ TEST FUNCTION ============ #
def test_agent(agent, test_maze):
    env = MazeEnv(test_maze)
    state = env.reset()
    path = [state]
    total_steps = 0

    print("Agent's movements on the test maze:")
    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.act()
        next_state, done, action_name = env.step(action)
        path.append(next_state)
        total_steps += 1

        print(f"Step {step + 1}: Position {next_state}, Action {action_name}, Goal Reached: {done}")

        if done:
            print(f"Goal reached in {step + 1} steps.")
            break
    else:
        print("Did not reach the goal within the maximum steps.")

    visualize_agent_run(env, path, total_steps)

def test_agent_multiple_runs(agent, test_maze, num_runs=10):
    """
    Test the random agent on the test maze multiple times and collect all paths.
    """
    env = MazeEnv(test_maze)
    all_positions = []  # Store paths from all runs

    for run in range(num_runs):
        state = env.reset()
        path = [state]  # Track positions for this run

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act()
            next_state, done, _ = env.step(action)
            path.append(next_state)
            if done:
                break

        all_positions.append(path)  # Add this run's path to the collection

    return all_positions

# ============ MAIN SCRIPT ============ #
ALL_MAZES = generate_mazes(SEED)
TEST_MAZE = ALL_MAZES[-1]

agent = RandomAgent(action_size=4)

print("\nTesting random agent on unseen test maze...")
#test_agent(agent, TEST_MAZE)

runs_positions = test_agent_multiple_runs(agent, TEST_MAZE, num_runs=NUM_TEST_RUNS)
visualize_multiple_runs(env=MazeEnv(TEST_MAZE), all_positions=runs_positions)
