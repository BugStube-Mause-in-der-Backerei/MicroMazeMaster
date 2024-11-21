from micromazemaster.models.maze import Maze
import random
from collections import Counter
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation

# ============ PARAMETERS ============ #
MAZE_SIZE = (10, 5)
SEED = 200
NUM_MAZES = 100
STARTING_POSITION = (0.5, 0.5)
GOAL_POSITION = (9.5, 4.5)
MAX_STEPS_PER_EPISODE = 500

# ============ ENVIRONMENT LOADING ============ #
def generate_mazes(seed):
    random.seed(seed)
    mazes = []
    for index in range(NUM_MAZES):
        mazes.append(Maze(MAZE_SIZE[0], MAZE_SIZE[1], seed=random.randint(1, 1000)))
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
        return not self.maze.is_valid_move(position, new_position)
    
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
def visualize_agent_run(env, positions, total_steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    margin = 0.5
    ax.set_xlim(-margin, env.width + margin)
    ax.set_ylim(-margin, env.height + margin)
    ax.set_aspect('equal', adjustable='box')

    for wall in env.walls:
        (x1, y1), (x2, y2) = wall.start_position, wall.end_position
        ax.plot([x1, x2], [y1, y2], 'k', linewidth=3)

    ax.plot(env.start_position[0], env.start_position[1], 'go', markersize=10, label="Start")
    ax.plot(env.goal_position[0], env.goal_position[1], 'ro', markersize=10, label="Goal")

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

    agent_dot, = ax.plot([], [], 'bo', markersize=8, label="Agent")

    def update(frame):
        agent_dot.set_data([positions[frame][0]], [positions[frame][1]])
        return agent_dot,

    ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=10, blit=True)

    ax.axis('off')
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.5, -1.5, f"Steps: {total_steps}", fontsize=10, transform=ax.transAxes)
    plt.show()

# ============ RANDOM AGENT CLASS ============ #
class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self):
        return random.choice(range(self.action_size))

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

# ============ MAIN SCRIPT ============ #
ALL_MAZES = generate_mazes(SEED)
TEST_MAZE = ALL_MAZES[-1]

agent = RandomAgent(action_size=4)

print("\nTesting random agent on unseen test maze...")
test_agent(agent, TEST_MAZE)
