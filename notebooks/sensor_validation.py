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

from micromazemaster.models.maze import Maze, Orientation
from micromazemaster.models.mouse import Mouse
from micromazemaster.utils.logging import logger

# +
maze = Maze(width=5, height=5, seed=950131)

maze.plot()

# +
actions = [
    "turn_right",
    "forward",
    "forward",
    "turn_left",
    "forward",
    "turn_left",
    "forward",
    "forward",
    "turn_right",
    "forward",
    "forward",
    "turn_right",
    "forward",
    "turn_left",
    "forward",
    "turn_right",
    "forward",
    "forward",
    "turn_right",
    "forward",
    "turn_left",
    "forward",
    "turn_right",
    "forward",
    "turn_right",
    "forward",
    "turn_left",
    "forward",
    "forward",
    "turn_left",
    "forward",
    "turn_left",
    "forward",
]

mouse = Mouse(x=0.5, y=0.5, orientation=Orientation.NORTH, maze=maze)

predicted_sensor_left = []
predicted_sensor_front = []
predicted_sensor_right = []

actual_sensor_left = [
    119.00,
    136.00,
    150.00,
    113.00,
    180.00,
    99.00,
    125.00,
    128.00,
    128.00,
    116.00,
    122.00,
    125.00,
    144.00,
    106.00,
    226.00,
    123.00,
    145.00,
    141.00,
    124.00,
    135.00,
    100.00,
    215.00,
    110.00,
    129.00,
    132.00,
    135.00,
    139.00,
    159.00,
    150.00,
    105.00,
    219.00,
    132.00,
    171.00,
    132.00,
]
actual_sensor_front = [
    67.00,
    229.00,
    237.00,
    71.00,
    245.00,
    99.00,
    227.00,
    228.00,
    65.00,
    223.00,
    225.00,
    68.00,
    241.00,
    96.00,
    255.00,
    96.00,
    234.00,
    232.00,
    67.00,
    250.00,
    83.00,
    252.00,
    91.00,
    229.00,
    75.00,
    240.00,
    243.00,
    231.00,
    233.00,
    108.00,
    240.00,
    80.00,
    232.00,
    84.00,
]
actual_sensor_right = [
    92.00,
    131.00,
    142.00,
    100.00,
    149.00,
    100.00,
    147.00,
    159.00,
    84.00,
    154.00,
    154.00,
    88.00,
    149.00,
    99.00,
    153.00,
    92.00,
    153.00,
    144.00,
    87.00,
    196.00,
    90.00,
    167.00,
    92.00,
    174.00,
    85.00,
    152.00,
    165.00,
    137.00,
    133.00,
    101.00,
    134.00,
    102.00,
    147.00,
    111.00,
]


def get_sensor_outputs():
    mouse.update_sensor()
    sensor_outputs = mouse.distance

    predicted_sensor_left.append(sensor_outputs[0])
    predicted_sensor_front.append(sensor_outputs[1])
    predicted_sensor_right.append(sensor_outputs[2])


get_sensor_outputs()

for action in actions:
    match action:
        case "forward":
            mouse.move_forward()
        case "turn_left":
            mouse.turn_left()
        case "turn_right":
            mouse.turn_right()
        case _:
            logger.error(f"Unknown action {action}")

    get_sensor_outputs()

# +
predicted_sensor_left = [round(x, 0) for x in predicted_sensor_left]
predicted_sensor_front = [round(x, 0) for x in predicted_sensor_front]
predicted_sensor_right = [round(x, 0) for x in predicted_sensor_right]

logger.info(f"Predicted sensor left: {predicted_sensor_left}")
logger.info(f"Acutal sensor left: {actual_sensor_left}\n")
logger.info(f"Predicted sensor front: {predicted_sensor_front}")
logger.info(f"Acutal sensor front: {actual_sensor_front}\n")
logger.info(f"Predicted sensor right: {predicted_sensor_right}")
logger.info(f"Acutal sensor right: {actual_sensor_right}\n")

# +
# Calculate the mean squared error

mse_left = sum([(x - y) ** 2 for x, y in zip(actual_sensor_left, predicted_sensor_left)]) / len(actual_sensor_left)
mse_front = sum([(x - y) ** 2 for x, y in zip(actual_sensor_front, predicted_sensor_front)]) / len(actual_sensor_front)
mse_right = sum([(x - y) ** 2 for x, y in zip(actual_sensor_right, predicted_sensor_right)]) / len(actual_sensor_right)

fig, axs = plt.subplots(1, 3, figsize=(25, 5))
fig.suptitle("Sensor Outputs")
axs[0].plot(actual_sensor_left, label="Actual")
axs[0].plot(predicted_sensor_left, label="Predicted")
axs[0].set_title("Left Sensor (MSE: {:.2f})".format(mse_left))
axs[0].legend()

axs[1].plot(actual_sensor_front, label="Actual")
axs[1].plot(predicted_sensor_front, label="Predicted")
axs[1].set_title("Front Sensor (MSE: {:.2f})".format(mse_front))
axs[1].legend()
axs[2].plot(actual_sensor_right, label="Actual")
axs[2].plot(predicted_sensor_right, label="Predicted")
axs[2].set_title("Right Sensor (MSE: {:.2f})".format(mse_right))
axs[2].legend()

plt.show()
