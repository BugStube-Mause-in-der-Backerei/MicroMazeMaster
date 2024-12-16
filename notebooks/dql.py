from micromazemaster.models.maze import Maze
import json
import numpy as np
from shapely.geometry import LineString
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation
from collections import deque, defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
import math

# ============ PARAMETER ============ #
MAZE_SIZE = (10, 5)
SEED = 20
ACTION_SEED = 5
NUM_MAZES = 20
NUM_AGENTS = 3
NUM_TEST_RUNS = 100
STARTING_POSITION = (0.5, 0.5)
GOAL_POSITION = (9.5, 4.5)

BATCH_SIZE = 32
HIDDEN_SIZE = 24

TRAINING_EPISODES = 10
LEARNING_RATE = 0.01
GAMMA = 1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
SEQUENCE_LENGTH = 10
MAX_STEPS_PER_EPISODE = 1500

REWARD_GOAL = 100000
REWARD_NEW_POSITION = 100
REWARD_DISTANCE_CHANGE = 10

PENALTY_STALL = -20
PENALTY_WALL_COLLISION = -10
PENALTY_REPETITIVE_ACTION = -50
PENALTY_FAILED_ACTION = -100

# ============ ENVIRONMENT LOADING ============ #

action_random = random.Random(ACTION_SEED)
torch.manual_seed(1)

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
        self.shapely_walls = [LineString(wall.get_positions()) for wall in self.walls]
        self.failed_actions = defaultdict(set)  # Track failed actions at each cell
        self.reset()

    def reset(self):
        """Reset environment to the starting position and initialize visited positions and recent history."""
        self.position = self.start_position
        self.done = False
        self.visited_positions = {}  # Track visited positions
        self.failed_actions.clear()  # Clear failed actions tracking
        return self.position, self.find_openings(self.position)

    def valid_move(self, position, new_position):
        """Check if a movement would hit a wall, using sorted lists of walls for faster searching."""
        return self.maze.is_valid_move(position, new_position)
    
    def distance_to_goal(self, position):
        """Calculate the Euclidean distance to the goal."""
        return math.sqrt((position[0] - self.goal_position[0]) ** 2 + (position[1] - self.goal_position[1]) ** 2)

    def is_valid_position(self, position):
        """Ensure the position is within maze boundaries."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def find_openings(self, position):
        return [
            self.maze.is_valid_move(position, (position[0], position[1] + 1)),
            self.maze.is_valid_move(position, (position[0] + 1, position[1])),
            self.maze.is_valid_move(position, (position[0], position[1] - 1)),
            self.maze.is_valid_move(position, (position[0] - 1, position[1]))
        ]

    def step(self, action):
        """Move the agent based on action and return position, reward, and completion status."""
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        action_names = ["North", "East", "South", "West"]
        dx, dy = moves[action]
        new_position = (self.position[0] + dx, self.position[1] + dy)

        available_actions = [i for i, val in enumerate(self.position) if val]

        reward = 0

        # Check for wall collision
        if self.valid_move(self.position, new_position):
            prev_distance = self.distance_to_goal(self.position)
            self.position = new_position
            new_distance = self.distance_to_goal(self.position)
            
            # Reward based on progress toward the goal
            distance_change = prev_distance - new_distance
            reward += REWARD_DISTANCE_CHANGE * distance_change
        else:
            reward += PENALTY_WALL_COLLISION  # Penalty for attempting to go through a wall
            self.failed_actions[self.position].add(action)  # Record failed action
            # Penalize repeated failed actions
            if action in self.failed_actions[self.position]:
                reward += PENALTY_FAILED_ACTION

        # Determine reward based on outcome of the move
        if self.position == self.goal_position:
            reward += REWARD_GOAL - (self.visited_positions.get(self.position, 0) * 10)  # Larger reward for faster reach
            self.done = True
        else:
            # Penalize repeating the same step (stuck behavior)
            if action in self.failed_actions[self.position]:
                reward += PENALTY_REPETITIVE_ACTION
            
            # Penalize stalling or revisiting recent positions
            if self.visited_positions.get(self.position, 0) > 2:
                reward += PENALTY_STALL
            else:
                # Encourage exploring new positions
                reward += REWARD_NEW_POSITION
                self.visited_positions[self.position] = self.visited_positions.get(self.position, 0) + 1

        return self.position, reward, self.done, action_names[action], self.find_openings(self.position)
    
# ============ VISUALIZATION ============ #
def visualize_agent_run(env, positions, total_reward, total_steps, caption="DQL auf ungesehenes Labyrinth"):
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
    side_legend_text = f"Steps: {total_steps}\nReward: {int(total_reward)}"
    fig.text(0.05, 0.79, side_legend_text, fontsize=10, va='center', ha='left', 
         bbox=dict(facecolor='white', alpha=0.8), transform=fig.transFigure)

    # Add caption above the plot, if provided
    if caption:
        fig.suptitle(caption, fontsize=12, weight='bold', y=0.95)

    # Show the animation
    plt.show()

def visualize_multiple_runs(env, all_positions, num_colors=100, caption="Zusammengefasstes DQL auf ungesehenes Labyrinth"):
    # Flatten all positions into a single list
    aggregated_positions = [pos for positions in all_positions for pos in positions]
    position_counts = Counter(aggregated_positions)
    max_visits = max(position_counts.values())
    
    # Adjust the colormap to have 'num_colors' steps
    colors = plt.cm.Blues(np.linspace(0.01, 2, num_colors))  # Gradation from light to dark blue
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

# ============ DEEP Q-NETWORK CLASS ============ #
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=HIDDEN_SIZE):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 24)
        self.fc2 = nn.Linear(24, action_size)

    def forward(self, x):
        """Perform a forward pass through the DQN."""
        h0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(device)
        c0 = torch.zeros(1, x.size(0), HIDDEN_SIZE).to(device)
        x, _ = self.lstm(x, (h0, c0))
        x = torch.relu(self.fc1(x[:, -1, :]))
        return self.fc2(x)


# ============ AGENT CLASS ============ #
class Agent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.epsilon = 1.0
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

    def store_experience(self, state_seq, action, reward, next_state_seq, done):
        """Store experiences in memory for training."""
        state_seq = [np.array(state).reshape(-1) for state in state_seq]
        next_state_seq = [np.array(state).reshape(-1) for state in next_state_seq]
        state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
        next_state_seq = torch.FloatTensor(next_state_seq).unsqueeze(0).to(device)
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def act(self, state_seq, env):
        """Choose action based on epsilon-greedy policy, avoiding previously failed actions."""
        if action_random.random() <= self.epsilon:
            available_actions = [i for i, val in enumerate(env.find_openings(env.position)) if val]
            if not available_actions:  # If all actions failed, allow any
                available_actions = list(range(self.action_size))
            return action_random.choice(available_actions)

        state_seq = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            best_action = torch.argmax(self.model(state_seq)).item()
        return best_action

    def replay(self):
        """Train model on randomly sampled experiences from memory."""
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a random minibatch from the replay memory
        minibatch = action_random.sample(self.memory, BATCH_SIZE)

        # Prepare arrays to batch process states, actions, rewards, etc.
        state_batch = torch.cat([experience[0] for experience in minibatch]).to(device)
        action_batch = torch.LongTensor([experience[1] for experience in minibatch]).to(device)
        reward_batch = torch.FloatTensor([experience[2] for experience in minibatch]).to(device)
        next_state_batch = torch.cat([experience[3] for experience in minibatch]).to(device)
        done_batch = torch.FloatTensor([float(experience[4]) for experience in minibatch]).to(device)

        # Compute Q-values for current states
        q_values = self.model(state_batch)  # Shape: [BATCH_SIZE, action_size]

        # Compute target Q-values for next states
        with torch.no_grad():
            next_q_values = self.model(next_state_batch)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]  # Take max over actions

        # Calculate target for each sample
        targets = reward_batch + GAMMA * max_next_q_values * (1 - done_batch)  # Zero out next Q for terminal states

        # Use advanced indexing to update only the Q-values corresponding to the chosen actions
        q_values_for_actions = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()

        # Calculate loss
        loss = self.loss_fn(q_values_for_actions, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon (for exploration vs. exploitation balance)
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY


    def clone_from(self, best_agent):
        """Copy the model and optimizer state from the best agent."""
        self.model.load_state_dict(best_agent.model.state_dict())
        self.optimizer.load_state_dict(best_agent.optimizer.state_dict())
        self.epsilon = best_agent.epsilon

# ============ TRAINING AND TESTING FUNCTIONS ============ #
def train_agents_on_maze(agents, maze_data, training_repeats=10):
    """Train agents on a single maze and clone the best-performing agent."""
    best_agent = None
    highest_reward = float('-inf')
    for i, agent in enumerate(agents):
        total_reward = 0
        env = MazeEnv(maze_data)
        state_seq = [env.reset()] * SEQUENCE_LENGTH
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state_seq, env)
            next_state, reward, done, _, openings = env.step(action)
            next_state_seq = state_seq[1:] + [(next_state, openings)]
            agent.store_experience(state_seq, action, reward, next_state_seq, done)
            state_seq = next_state_seq
            total_reward += reward
            if done:
                break
        agent.replay()
        #print(f"Agent {i + 1} completed training on maze with total reward: {total_reward}")
        if total_reward > highest_reward:
            highest_reward = total_reward
            best_agent = agent
    #print(f"Best agent selected with reward: {highest_reward}")
    return best_agent

def train_agents_across_mazes(agents, mazes, training_episodes=TRAINING_EPISODES):
    """Train agents across multiple mazes for a specified number of episodes."""
    for episode in range(training_episodes):
        for maze_index, maze in enumerate(mazes):
            #print(f"  Training agents on maze {maze_index + 1}/{len(mazes)}...")
            best_agent = train_agents_on_maze(agents, maze)
            for agent in agents:
                agent.clone_from(best_agent)

def test_agent_multiple_runs(agent, test_maze, num_runs=10):
    """Test the trained agent on the test maze multiple times and visualize the aggregated paths."""
    env = MazeEnv(test_maze)
    all_positions = []  # To store paths from multiple runs

    for run in range(num_runs):
        state_seq = [env.reset()] * SEQUENCE_LENGTH
        positions = [env.position]  # Track positions for this run

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state_seq, env)
            next_state, _, done, _, openings = env.step(action)
            state_seq = state_seq[1:] + [next_state, openings]
            positions.append(next_state)
            if done:
                break

        all_positions.append(positions)  # Store the path of this run

    # Visualize the aggregated runs
    visualize_multiple_runs(env, all_positions)

def test_agent(agent, test_maze):
    """Test the trained agent on an unseen maze and render the path step-by-step."""
    env = MazeEnv(test_maze)
    state_seq = [env.reset()] * SEQUENCE_LENGTH
    path = [env.position]  # Start tracking the path from the initial position

    total_reward = 0  # Track total reward during testing
    positions = [env.position]  # List to store positions for animation

    #print("Agent's movements on the test maze:")
    for step in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state_seq, env)
        next_state, reward, done, action_name, openings = env.step(action)
        state_seq = state_seq[1:] + [(next_state, openings)]
        path.append(next_state)  # Track the new position
        positions.append(next_state)  # Append to positions for animation
        total_reward += reward  # Update total reward

        # Print the current position, action, reward, and whether the goal was reached
        #print(f"Step {step + 1}: Position {next_state}, Action {action_name}, Reward {reward}, Goal Reached: {done}")

        if done:
            print(f"Goal reached in {step + 1} steps.")
            break
    else:
        print("Did not reach the goal within the maximum steps.")

    # Visualize the agent's movements step-by-step
    visualize_agent_run(env, positions, total_reward, len(positions) - 1)


# ============ MAIN SCRIPT ============ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mazes and split into training and testing sets
ALL_MAZES = generate_mazes(SEED)
TRAINING_MAZES = ALL_MAZES[:-1]
TEST_MAZE = ALL_MAZES[-1]

# Initialize multiple agents
agents = [Agent(state_size=6, action_size=4) for _ in range(NUM_AGENTS)]

# Train across mazes
print("Training agents across mazes...")
train_agents_across_mazes(agents, TRAINING_MAZES, TRAINING_EPISODES)

# Test the best agent from the last maze training
best_agent = agents[0]  # Any agent, as they are all cloned to the best one at the end
print("\nTesting the best agent on unseen test maze...")
#test_agent(best_agent, TEST_MAZE)
test_agent_multiple_runs(best_agent, TEST_MAZE, num_runs=NUM_TEST_RUNS)

