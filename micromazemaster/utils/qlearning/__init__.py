import numpy as np
from micromazemaster.utils.config import settings
from micromazemaster.utils.qlearning.helper import (
    plot_perceived_walls,
    plot_training_progress,
    preprocess_walls,
    simulate_path,
    train,
)


class Qlearning:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.start_position = start_position
        self.goal_position = goal_position

    def run(self):
        maze_size = (self.maze.width, self.maze.height)

        q_table = {}
        for x in np.arange(0.5, maze_size[0], 1):
            for y in np.arange(0.5, maze_size[1], 1):
                for orientation in range(4):
                    q_table[(x, y, orientation)] = np.zeros(len(settings.ACTIONS))

        walls_sorted_by_x, walls_sorted_by_y = preprocess_walls(self.maze.walls)

        episode_rewards, episode_steps, best_path = train(
            num_episodes=settings.NUM_EPISODES,
            max_steps_per_episode=settings.MAX_STEPS_PER_EPISODE,
            q_table=q_table,
            actions=settings.ACTIONS,
            start_position=self.start_position,
            goal_position=self.goal_position,
            epsilon_decay=settings.EPSILON_DECAY,
            learning_rate=settings.LEARNING_RATE,
            discount_factor=settings.DISCOUNT_FACTOR,
            epsilon=settings.EPSILON_START,
            walls_sorted_by_x=walls_sorted_by_x,
            walls_sorted_by_y=walls_sorted_by_y,
            maze_size=maze_size,
        )

        actions_taken, path, perceived_walls = simulate_path(
            start_position=self.start_position,
            goal_position=self.goal_position,
            q_table=q_table,
            actions=settings.ACTIONS,
            walls_sorted_by_x=walls_sorted_by_x,
            walls_sorted_by_y=walls_sorted_by_y,
            max_steps=settings.MAX_STEPS_PER_EPISODE,
            maze_size=maze_size,
        )

        fig = plot_perceived_walls(
            walls=self.maze.walls,
            perceived_walls=perceived_walls,
            path=path,
            start_position=self.start_position,
            goal_position=self.goal_position,
            maze_size=maze_size,
        )

        plot_training_progress(episode_rewards=episode_rewards, episode_steps=episode_steps)
        return fig, path
