import sys, os

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools")) # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools")) # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
import numpy as np
from RLAgent import RLAgent
from collections import deque
from help_funcs import get_density

class TrafficSignalController(object):
    def __init__(self, _id, tsc_data, conn, args):
        self.id = _id
        self.tsc_data = tsc_data
        self.phase_time = 0
        self.conn = conn
        self.yellow_t = args.yellow_t
        self.red_t = args.red_t
        self.a_repeat = args.a_repeat
        self.current_phase = None
        self.args = args

    def next_phase_and_duration(self, local_obs):
        pass

    def update(self, local_obs):
        pass

    def run(self, local_obs):
        ###ultimately must either keep current phase (the no action, dont change) OR
        ###supply new phase (the change action)
        ###temporally it infinitely counts down -1 steps
        ###timer reset when new phase enacted

        self.update(local_obs)

        if self.phase_time == 0:
            ###get new phase and duration
            next_phase, duration = self.next_phase_and_duration(local_obs)
            self.phase_time = duration
            self.conn.trafficlight.setRedYellowGreenState( self.id, next_phase )
            self.current_phase = next_phase
        self.phase_time -= 1

class RLTrafficSignalController(TrafficSignalController):
    ###implements a cycle, fixed uniform phase duration for all green phases
    def __init__(self, _id, tsc_data, conn, args, exp_replay, neural_networks, eps, rl_stats, reward):
        super(RLTrafficSignalController, self).__init__(_id, tsc_data, conn, args)
        self.rlagent = RLAgent(neural_networks, eps, exp_replay, tsc_data['n_green_phases'], args.n_steps, args.batch, args.replay, args.gamma )
        ###set intersection to red default
        self.id = _id
        #self.phase_buffer = deque()
        self.exp = {}
        self.current_phase = tsc_data['all_red']
        self.args = args
        self.phase_deque = deque()
        self.state_deque = deque()
        self.acting = False
        self.rl_stats = rl_stats
        self.reward = reward
        self.rewards = []

    def update(self, local_obs):
        ###update state buffer
        state = get_density(local_obs, self.tsc_data['inc_lanes'], self.tsc_data['lane_lengths'], self.args.v_len)
        self.state_deque.append(state)

    def next_phase_and_duration(self, local_obs):
        if len(self.phase_deque) == 0:
            if self.acting == True:
                if self.args.mode == 'train':
                    next_s = self.observe_state()
                    r = self.get_reward()
                    terminal = True if np.sum(self.state_deque[-1]) == 0 else False
                    self.rlagent.store_experience(self.exp['s'], self.exp['a'], next_s, r, terminal)
                    self.rl_stats['n_exp'] += 1.0/self.args.n_steps
                    #if self.rl_stats['n_exp'] % 100 == 0:
                    #    print('exp replay size '+str(self.rl_stats['n_exp']))
                    #    print('updates '+str(self.rl_stats['updates']))

            if len(self.state_deque) == 0 or np.sum(self.state_deque[-1]) == 0:
                ###no vehicle present, default to all red
                self.phase_deque.append(( self.tsc_data['all_red'], 1))
                self.acting = False
            else:
                ###observe state 
                s = self.observe_state()
                self.exp['s'] = s 
                ###get new params before acting
                self.rlagent.set_params('online', self.rl_stats['online'])
                
                ##take action using rl agent
                s = s[np.newaxis, ...]
                action_idx = self.rlagent.get_action(s)
                self.exp['a'] = action_idx

                ###change action index to green traffic signal phase
                next_green = self.tsc_data['int_to_action'][action_idx] 
                self.acting = True
                                                                              
                ###add transition phases for desired duration
                transitions = get_transitions(self.current_phase, next_green)
                for trans in transitions:
                    if 'y' in trans:
                        t = self.yellow_t
                    else:
                        t = self.red_t
                    self.phase_deque.append((trans, t))
                self.phase_deque.append((next_green, self.a_repeat))
            
        next_phase_and_duration = self.phase_deque.popleft()
        next_phase = next_phase_and_duration[0]
        duration = next_phase_and_duration[1]

        return next_phase, duration

    def observe_state(self):                                                                 
        traffic_state = np.array(self.state_deque[-1])
        signal_state = np.array(self.tsc_data['phase_one_hot'][self.current_phase])
        s = np.concatenate([traffic_state, signal_state])
        #s = s[np.newaxis,...]
        return s

    def update_max_reward(self, r):
        abs_r = np.absolute(r)
        if abs_r > self.rl_stats['max_r']:
            self.rl_stats['max_r'] = abs_r

    def get_reward(self):
        r = -np.sum([ self.reward(e) for e in self.tsc_data['inc_edges'] ])
        #self.rewards.append(r)
        self.update_max_reward(r)
        return r

class UniformFixedTrafficSignalController(TrafficSignalController):
    ###implements a cycle, fixed uniform phase duration for all green phases
    def __init__(self, _id, tsc_data, conn, args, uniform):
        super(UniformFixedTrafficSignalController, self).__init__(_id, tsc_data, conn, args)
        ###mayber just make cycle here once and then iterate thru it??
        self.uniform = uniform
        self.phase_idx = 0
        ###self.cycle = \\function that creates cycle from sequence of greens using uniform time, need transitions
        self.cycle = gen_uniform_fixed_cycle(tsc_data['green_phases'], uniform, self.yellow_t, self.red_t)
        self.cycle_len = len(self.cycle)

    def next_phase_and_duration(self, local_obs):
        ###implement cycle of green with uniform length
        ###need logic here that outputs red and yellow phases as well
        ###just loop thru cycle, should include all phases, greenn, yellow and red
        next_phase_and_duration = self.cycle[self.phase_idx]
        self.phase_idx += 1
        ###reset cycle
        if self.phase_idx == self.cycle_len:
            self.phase_idx = 0
        return next_phase_and_duration[0], next_phase_and_duration[1]

    def update(self, local_obs):
        return None


class FixedTrafficSignalController(TrafficSignalController):
    ###implements a cycle, fixed phase durations but can be different from each other (can be set with Websters)
    def __init__(self, _id, tsc_data, conn, args, gtimes):
        super(FixedTrafficSignalController, self).__init__(_id, tsc_data, conn, args)
        ###mayber just make cycle here once and then iterate thru it??
        self.gtimes = gtimes
        self.phase_idx = 0
        ###self.cycle = \\function that creates cycle from sequence of greens using uniform time, need transitions
        self.cycle = gen_fixed_cycle(tsc_data['green_phases'], gtimes, self.yellow_t, self.red_t)
        self.cycle_len = len(self.cycle)

    def next_phase_and_duration(self, local_obs):
        ###implement cycle of green with uniform length
        ###need logic here that outputs red and yellow phases as well
        ###just loop thru cycle, should include all phases, greenn, yellow and red
        next_phase_and_duration = self.cycle[self.phase_idx]
        self.phase_idx += 1
        ###reset cycle
        if self.phase_idx == self.cycle_len:
            self.phase_idx = 0
        return next_phase_and_duration[0], next_phase_and_duration[1]

    def update(self, local_obs):
        return None

def gen_uniform_fixed_cycle(green_cycle, uniform, yellow_t, red_t):
    ###green cycle should be a list or the green actualy movement light phases tuples
    cycle = []
    next_phase = green_cycle+[green_cycle[0]]
    for g, g_  in  zip(green_cycle, next_phase[1:]):
        ###add greeen
        cycle.append((g, uniform))
        ###add yellow and red if necessary
        transitions = get_transitions(g, g_)
        ###transitions could be 0
        for trans in transitions:
            if 'y' in trans:
                t = yellow_t
            else:
                t = red_t
            cycle.append((trans, t))
    return cycle

def gen_fixed_cycle(green_cycle, gtimes, yellow_t, red_t):
    ###green cycle should be a list or the green actualy movement light phases tuples
    cycle = []
    next_phase = green_cycle+[green_cycle[0]]
    for g, g_, gt  in  zip(green_cycle, next_phase[1:], gtimes):
        ###add greeen
        cycle.append((g, gt))
        ###add yellow and red if necessary
        transitions = get_transitions(g, g_)
        ###transitions could be 0
        for trans in transitions:
            if 'y' in trans:
                t = yellow_t
            else:
                t = red_t
            cycle.append((trans, t))
    return cycle

def get_transitions(current_phase, next_phase):
    ###arguments must be same length
    y = None
    r = None
    for c, n in zip(current_phase, next_phase):
        if c == 'g' and n == 'G':
            ##advance turn, only need yellow
            ###add yellow only
            ###G - > r becomes y, else leave alone
            y = ''.join([ 'y' if C == 'G' and N == 'r' else C for C, N in zip(current_phase, next_phase) ])
            return [y]

    for c, n in zip(current_phase, next_phase):
        if ( c == 'g' and n == 'r' ) or ( c == 'G' and n == 'r' ):
            ##protected change, need all red stop
            ###red and yellow
            y = ''.join([ 'y'  if ( C == 'g' and N == 'r' ) or ( C == 'G' and N == 'r' ) else C for C, N in zip(current_phase, next_phase) ])
            ###all red
            r = 'r'*len(current_phase)
            return [y,r]

    return []
