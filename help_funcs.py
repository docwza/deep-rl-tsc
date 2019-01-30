import numpy as np
import pickle

def input_to_one_hot(phases):
    identity = np.identity(len(phases))
    one_hots = { phases[i]:identity[i,:]  for i in range(len(phases)) }
    return one_hots

def int_to_input(phases):
    return { p:phases[p] for p in range(len(phases)) }

def action_state_lanes( actions, index_to_lane):
    action_state_lanes = {a:[] for a in actions}
    for a in actions:
        for s in range(len(a)):
            if a[s] == 'g' or a[s] == 'G':
                action_state_lanes[a].append(index_to_lane[s])
        ###some movements are on the same lane, removes duplicate lanes
        action_state_lanes[a] = list(set(action_state_lanes[a]))

    return action_state_lanes

def get_density(lane_vehs, lanes, lane_lengths, v_len):
    density = np.array([float(len(lane_vehs[lane])) for lane in lanes])
    length_normalizer = np.array([ lane_lengths[lane] for lane in lanes])
    ###vehicles in lane/lane vehicle capacity
    density /= (length_normalizer/v_len)
    return density

def save_data(fp, data):
    with open(fp, "wb") as fo:
        pickle.dump(data, fo)

def load_data(fp):
    with open(fp, "rb") as fo:
        data = pickle.load(fo)
    return data
