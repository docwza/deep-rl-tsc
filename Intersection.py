from TrafficSignalController import RLTrafficSignalController, UniformFixedTrafficSignalController, FixedTrafficSignalController

class Intersection():
    def __init__(self, _id, tsc_type, conn, args, tsc_data, exp_replay=None, neural_networks=None, eps=None, rl_stats=None, reward = None):
        ###use _id and net data to create the tsc, pass it neural network and exp replay????
        if tsc_type == 'fixed':
            self.tsc = FixedTrafficSignalController(_id, tsc_data, conn, args, [15 for _ in range(tsc_data['n_green_phases'])])
        elif tsc_type == 'uniform':
            self.tsc = UniformFixedTrafficSignalController(_id, tsc_data, conn, args, 15 )
        elif tsc_type == 'rl':
            self.tsc = RLTrafficSignalController( _id, tsc_data, conn, args, exp_replay, neural_networks, eps, rl_stats, reward )

    def run(self, lane_vehicles):
        ###get traffic data, run traffic signal controller
        self.tsc.run(lane_vehicles)
