import random

import customtkinter as ctk

from micromazemaster.models.maze import Maze
from micromazemaster.utils.logging import logger
from micromazemaster.utils.qlearning import Qlearning

maze = None
path = None
scale = 50
dot_radius = 10


def button_callback():
    global maze
    global path
    maze = Maze(10, 10, random.randrange(100))
    model = Qlearning(maze=maze, start_position=(0.5, 0.5), goal_position=(5.5, 5.5))
    _plt, path = model.run()
    update_canvas()


def update_canvas():
    global path
    global maze
    if maze is not None:
        canvas.delete("all")

        for wall in maze.walls:
            x1, y1 = wall.start_position
            x2, y2 = wall.end_position
            canvas.create_line(x1 * scale + 2, y1 * scale + 2, x2 * scale + 2, y2 * scale + 2, width=2, fill="blue")

    if path is not None:
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            canvas.create_line(x1 * scale, y1 * scale, x2 * scale, y2 * scale, width=3, fill="red")

        x, y = path[0]
        canvas.create_oval(
            (x * scale) - dot_radius,
            (y * scale) - dot_radius,
            (x * scale) + dot_radius,
            (y * scale) + dot_radius,
            fill="green",
        )

        x, y = path[-1]
        canvas.create_oval(
            (x * scale) - dot_radius,
            (y * scale) - dot_radius,
            (x * scale) + dot_radius,
            (y * scale) + dot_radius,
            fill="red",
        )


def on_close():
    logger.info("Window is closing...")
    root.quit()


def micromazemaster_gui():
    global canvas
    global root

    root = ctk.CTk()
    root.geometry("600x500")

    canvas = ctk.CTkCanvas(master=root, width=500, height=500, bg="white")
    canvas.pack(padx=20, pady=20)

    button = ctk.CTkButton(root, text="Generate Maze", command=button_callback)
    button.pack(padx=20, pady=20)

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


if __name__ == "__main__":
    micromazemaster_gui()
