import bioviz
import matplotlib.pyplot as plt
import numpy as np

sol_name = ""
biorbd_model_path = "models/pendulum_maze.bioMod"
# sol_dir = "solutions/" + sol_name

b = bioviz.Viz(biorbd_model_path, background_color=(1, 1, 1), markers_size=0.04)
# b.load_movement()
b.exec()