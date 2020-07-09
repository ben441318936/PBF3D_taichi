import pickle
import numpy as np

iter_states_path = "./states/iter_states.obj"
with open(iter_states_path, "rb") as f:
    iter_states = pickle.load(f)

best_states_path = "./states/best_states.obj"
with open(best_states_path, "rb") as f:
    best_states = pickle.load(f)


print(len(iter_states))