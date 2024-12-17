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
    def __init__(self, maze, start_position, goal_position, close_figure=False):
        self.maze = maze
        self.start_position = start_position
        self.goal_position = goal_position
        self.close_figure = close_figure

    def run(self):
        maze_size = (self.maze.width, self.maze.height)

        q_table = {}
        for x in np.arange(0.5, maze_size[0], 1):
            for y in np.arange(0.5, maze_size[1], 1):
                for orientation in range(4):
                    q_table[(x, y, orientation)] = np.zeros(len(settings.ql.actions))

        walls_sorted_by_x, walls_sorted_by_y = preprocess_walls(self.maze.walls)

        episode_rewards, episode_steps, _ = train(
            num_episodes=settings.ql.num_episodes,
            max_steps_per_episode=settings.ql.max_steps_per_episode,
            q_table=q_table,
            actions=settings.ql.actions,
            start_position=self.start_position,
            goal_position=self.goal_position,
            epsilon_decay=settings.ql.epsilon_decay,
            learning_rate=settings.ql.learning_rate,
            discount_factor=settings.ql.discount_factor,
            epsilon=settings.ql.epsilon_start,
            walls_sorted_by_x=walls_sorted_by_x,
            walls_sorted_by_y=walls_sorted_by_y,
            maze_size=maze_size,
        )

        _, path, perceived_walls = simulate_path(
            start_position=self.start_position,
            goal_position=self.goal_position,
            q_table=q_table,
            actions=settings.ql.actions,
            walls_sorted_by_x=walls_sorted_by_x,
            walls_sorted_by_y=walls_sorted_by_y,
            max_steps=settings.ql.max_steps_per_episode,
            maze_size=maze_size,
        )

        fig = plot_perceived_walls(
            walls=self.maze.walls,
            perceived_walls=perceived_walls,
            path=path,
            start_position=self.start_position,
            goal_position=self.goal_position,
            maze_size=maze_size,
            close_figure=self.close_figure,
        )

        plot_training_progress(episode_rewards=episode_rewards, episode_steps=episode_steps)
        return fig, path
