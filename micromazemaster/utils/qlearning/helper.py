import random

import matplotlib.pyplot as plt
import numpy as np
from micromazemaster.utils.config import settings
from micromazemaster.utils.logging import logger


def preprocess_walls(walls):
    walls_as_tuples = [wall.get_positions() for wall in walls]

    walls_sorted_by_x = sorted(
        walls_as_tuples, key=lambda wall: (min(wall[0][0], wall[1][0]), min(wall[0][1], wall[1][1]))
    )

    walls_sorted_by_y = sorted(
        walls_as_tuples, key=lambda wall: (min(wall[0][1], wall[1][1]), min(wall[0][0], wall[1][0]))
    )

    return walls_sorted_by_x, walls_sorted_by_y


def hits_wall(position, new_position, walls_sorted_by_x, walls_sorted_by_y):
    x, y = position
    nx, ny = new_position

    curr_x, curr_y = int(x), int(y)

    if ny > y:
        relevant_walls = walls_sorted_by_y
        movement_direction = "north"
    elif ny < y:
        relevant_walls = walls_sorted_by_y
        movement_direction = "south"
    elif nx > x:
        relevant_walls = walls_sorted_by_x
        movement_direction = "east"
    elif nx < x:
        relevant_walls = walls_sorted_by_x
        movement_direction = "west"
    else:
        return False

    if movement_direction in ["north", "south"]:
        for wall in relevant_walls:
            (wx1, wy1), (wx2, wy2) = sorted(wall)
            if wy1 == wy2:
                if movement_direction == "north" and wy1 == curr_y + 1 and wx1 <= curr_x < wx2:
                    return True
                elif movement_direction == "south" and wy1 == curr_y and wx1 <= curr_x < wx2:
                    return True
    else:
        for wall in relevant_walls:
            (wx1, wy1), (wx2, wy2) = sorted(wall)
            if wx1 == wx2:
                if movement_direction == "east" and wx1 == curr_x + 1 and wy1 <= curr_y < wy2:
                    return True
                elif movement_direction == "west" and wx1 == curr_x and wy1 <= curr_y < wy2:
                    return True
    return False


def is_valid_position(position, maze_size):
    x, y = position
    return 0 <= x < maze_size[0] and 0 <= y < maze_size[1]


def get_next_position(position, orientation, action, walls_sorted_by_x, walls_sorted_by_y, maze_size):
    x, y = position

    if action == "turn_left":
        new_orientation = (orientation - 1) % 4
        return (x, y), new_orientation
    elif action == "turn_right":
        new_orientation = (orientation + 1) % 4
        return (x, y), new_orientation
    elif action == "forward":
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = moves[orientation]
        new_x, new_y = x + dx, y + dy

        new_position = (new_x, new_y)
        if hits_wall(position, new_position, walls_sorted_by_x, walls_sorted_by_y):
            return position, orientation

        if not is_valid_position(new_position, maze_size):
            logger.error(f"Invalid position detected: {new_position}")
            return position, orientation

        return new_position, orientation

    return position, orientation


def train(
    num_episodes,
    max_steps_per_episode,
    start_position,
    actions,
    q_table,
    goal_position,
    learning_rate,
    discount_factor,
    epsilon_decay,
    epsilon,
    walls_sorted_by_x,
    walls_sorted_by_y,
    maze_size,
):
    best_path_length = float("inf")
    best_path = None
    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        position = start_position
        orientation = 0
        prev_positions = [position]
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps_per_episode:
            state = (position[0], position[1], orientation)

            if random.uniform(0, 1) < epsilon:
                action = random.choice(settings.actions)
            else:
                action = actions[np.argmax(q_table[state])]

            new_position, new_orientation = get_next_position(
                position, orientation, action, walls_sorted_by_x, walls_sorted_by_y, maze_size
            )

            next_state = (new_position[0], new_position[1], new_orientation)

            if next_state not in q_table:
                logger.warning(f"Invalid state reached: {next_state}. Skipping this step.")
                continue

            hit_wall = new_position == position and action == "forward"

            reward = get_reward(new_position, goal_position, hit_wall, prev_positions)
            total_reward += reward

            current_q = q_table[state][actions.index(action)]
            max_future_q = np.max(q_table[next_state])
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
            q_table[state][actions.index(action)] = new_q

            position = new_position
            orientation = new_orientation
            prev_positions.append(position)
            if action == "forward":
                steps += 1

            if position == goal_position:
                done = True

        if position == goal_position and len(prev_positions) < best_path_length:
            best_path_length = len(prev_positions)
            best_path = prev_positions.copy()
            logger.info(f"New best path found in episode {episode + 1}!")
            logger.info(f"Path length: {best_path_length}")

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        epsilon = max(0.1, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            logger.info(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Steps: {steps}, Total Reward: {total_reward:.1f}, "
                f"Epsilon: {epsilon:.3f}"
            )

    return episode_rewards, episode_steps, best_path


def get_reward(position, goal, hit_wall, prev_positions):
    if position == goal:
        return 200
    if hit_wall:
        return -100
    if position in prev_positions[:-1]:
        return -50

    distance_to_goal = np.sqrt((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)
    distance_reward = -0.05 * distance_to_goal

    return distance_reward


def plot_perceived_walls(walls, perceived_walls, path, start_position, goal_position, maze_size, close_figure=False):
    fig, ax = plt.subplots(figsize=maze_size)

    for wall in walls:
        (x1, y1), (x2, y2) = wall.get_positions()
        ax.plot([x1, x2], [y1, y2], "k", linewidth=2, zorder=1)

    for wall in perceived_walls:
        (x1, y1), (x2, y2) = wall
        ax.plot([x1, x2], [y1, y2], linestyle="--", linewidth=1, color="red", zorder=2, label="Perceived Wall")

    path_x, path_y = zip(*path)
    ax.plot(path_x, path_y, "b-", label="Path", zorder=3)

    ax.plot(start_position[0], start_position[1], "go", markersize=10, label="Start", zorder=4)
    ax.plot(goal_position[0], goal_position[1], "ro", markersize=10, label="Goal", zorder=4)

    ax.set_xlim(-1, maze_size[0])
    ax.set_ylim(-1, maze_size[1])
    ax.grid(True)

    ax.legend()
    ax.set_title("Maze with Perceived Walls")

    if close_figure:
        plt.close(fig)

    return fig


def simulate_path(
    start_position, goal_position, q_table, max_steps, actions, walls_sorted_by_x, walls_sorted_by_y, maze_size
):
    position = start_position
    orientation = 0
    actions_taken = []
    visited_positions = [position]
    steps = 0
    perceived_walls = []

    while position != goal_position and steps < max_steps:
        state = (position[0], position[1], orientation)

        if state in q_table:
            action_index = np.argmax(q_table[state])
            action = actions[action_index]
        else:
            action = random.choice(actions)

        actions_taken.append(action)

        new_position, new_orientation = get_next_position(
            position, orientation, action, walls_sorted_by_x, walls_sorted_by_y, maze_size
        )

        if not is_valid_position(new_position, maze_size) or (
            action == "forward" and hits_wall(position, new_position, walls_sorted_by_x, walls_sorted_by_y)
        ):
            perceived_walls.append((position, new_position))
            new_position = position
        else:
            position, orientation = new_position, new_orientation
            visited_positions.append(position)

        steps += 1

    return actions_taken, visited_positions, perceived_walls


def plot_training_progress(episode_rewards, episode_steps):

    episodes = range(1, len(episode_rewards) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(episodes, episode_rewards, label="Total Reward", color="b")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episodes, episode_steps, label="Steps to Goal", color="g")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps to Reach Goal per Episode")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()