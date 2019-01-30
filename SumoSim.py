import sys, os, subprocess, time

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools")) # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools")) # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
from Vehicles import Vehicles
from NetworkData import NetworkData
from Intersection import Intersection
from help_funcs import input_to_one_hot, int_to_input, action_state_lanes 
import numpy as np

class SumoSim():
    def __init__(self, port, idx, cfg_fp):
        self.port = port
        self.idx = idx
        self.cfg_fp = cfg_fp

    def gen_sim(self, nogui, sim_len):
        if nogui is False:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        ###start thread with sumo sim
        self.sumo_process = subprocess.Popen([sumoBinary, "-c", self.cfg_fp, "--remote-port", str(self.port), "--no-warnings", "--no-step-log", "--random"], stdout=None, stderr=None)
        ###connect to sumo sim through specific port
        self.conn = traci.connect(self.port)
        self.sim_len = sim_len
        self.t = 0

    def get_tsc_data(self):
        tsc = self.conn.trafficlight.getIDList()
        tsc_data = { _id:{} for _id in tsc }
        

        for tsc in tsc_data:
            #print('----- '+tl)                                                                                            
            ###get green phases for TL
            tsc_logic =  self.conn.trafficlight.getCompleteRedYellowGreenDefinition(tsc)[0]
            ##http://www.sumo.dlr.de/pydoc/traci._trafficlight.html#TrafficLightDomain-setCompleteRedYellowGreenDefinition
            ### getPhases(self) should work on tl_logic but it doesnt, so I have to scrape the phases like the following
            phases = []
            ###this code works for sumo version <= 0.32
            '''
            for phase in tl_logic._phases:
                if ('g' in phase._phaseDef or 'G' in phase._phaseDef) and 'y' not in phase._phaseDef:
                    phases.append(phase._phaseDef)
            '''
            for p in tsc_logic.getPhases():
                #print(p.state)
                if ('g' in p.state or 'G' in p.state ) and 'y' not in p.state:
                    phases.append(p.state)

            tsc_data[tsc]['green_phases'] = phases
            tsc_data[tsc]['n_green_phases'] = len(phases)
            tsc_data[tsc]['all_red'] = 'r'*len(phases[0])
              
            lanes = self.conn.trafficlight.getControlledLanes(tsc)
            ###incoming lanes 
            tsc_data[tsc]['inc_lanes'] = list(set(lanes))
            index_to_lane = { i:lane  for lane, i in zip(lanes, range(len(lanes))) }
            tsc_data[tsc]['phase_lanes'] = action_state_lanes( tsc_data[tsc]['green_phases'], index_to_lane )
            tsc_data[tsc]['inc_edges'] = set([ self.conn.lane.getEdgeID(l) for l in lanes])
            phases = sorted(phases)
            ##get one hot for actions and phases for state
            tsc_data[tsc]['action_one_hot'] = input_to_one_hot(phases)
            tsc_data[tsc]['int_to_action'] = phases
            tsc_data[tsc]['phase_one_hot'] = input_to_one_hot(phases+[tsc_data[tsc]['all_red']])
            tsc_data[tsc]['int_to_phase'] = int_to_input(phases+[tsc_data[tsc]['all_red']])
            ###lane lengths for normalization
            tsc_data[tsc]['lane_lengths'] = { l:self.conn.lane.getLength(l) for l in lanes }

        return tsc_data

    def run(self, net_data, args, exp_replay, neural_network, eps, rl_stats ):
        ###for batch vehicle data, faster than API calls
        data_constants = [traci.constants.VAR_SPEED, traci.constants.VAR_POSITION , traci.constants.VAR_LANE_ID, traci.constants.VAR_LANE_INDEX]
        self.vehicles = Vehicles(self.conn, data_constants, net_data, self.sim_len, args.demand, args.scale)

        ###create some intersections
        intersections = [ Intersection(_id, args.tsc, self.conn, args, net_data['tsc'][_id], exp_replay[_id], neural_network[_id], eps, rl_stats[_id], self.vehicles.get_edge_delay ) for _id in self.conn.trafficlight.getIDList()]

        print('start running sumo sim on port '+str(self.port))
        while self.t < self.sim_len:
            ###loop thru tsc intersections
            ###run and pass v_data
            lane_vehicles = self.vehicles.run()
            for i in intersections:
                i.run(lane_vehicles)
            self.step()

        if args.mode == 'train':
            ###write travel time mean to csv for graphing after training
            self.write_csv(str(eps)+'.csv', np.mean([self.vehicles.travel_times[v] for v in self.vehicles.travel_times]) )
        '''
        if eps < 0.1:
        #    #print('--- SIM FINISHED -- '+str(eps)+' average tt '+str(np.mean([self.vehicles.travel_times[v] for v in self.vehicles.travel_times])) +' std tt '+str(np.std([self.vehicles.travel_times[v] for v in self.vehicles.travel_times]))   )
            self.write_csv('hpresults.csv', np.mean([self.vehicles.travel_times[v] for v in self.vehicles.travel_times]) )

        '''
        #self.write_csv(str(eps)+'.csv', np.sum([i.tsc.rewards for i in intersections]) )
        #print('-------______TT -------------')
        #print(np.mean([self.vehicles.travel_times[v] for v in self.vehicles.travel_times]))

        print('finished running sumo sim on port '+str(self.port))
        self.cleanup()

    def write_csv(self, fp, data):
        with open(fp, 'a+') as f:
            f.write(str(data)+'\n')

    def step(self):
        self.conn.simulationStep()
        self.t += 1

    def cleanup(self):
        self.conn.close()
        self.sumo_process.terminate()
        print('finished cleaning up sim on port '+str(self.port)+' after '+str(self.t)+' steps')

if __name__ == '__main__':

    start_t = time.time()
    net_fp = 'net.net.xml'
    network_data = NetworkData(net_fp)

    print('network time elapsed '+str(time.time()-start_t))

    port = 9000
    idx = 0
    cfg_fp = 'net.sumocfg'

    sim = SumoSim(port, idx, cfg_fp)
    sim_len = 86400
    nogui = True
    sim.gen_sim(nogui, sim_len)
    tsc_data  = sim.get_tsc_data()
    for tsc in tsc_data:
        print(tsc)
        print(tsc_data[tsc])
    sim.run()
