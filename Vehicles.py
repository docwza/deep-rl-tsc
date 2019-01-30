import os, sys
# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools")) # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools")) # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

###now can import SUMO traci module
import traci
import numpy as np
from NetworkData import NetworkData

class Vehicles():
    def __init__(self, conn, constants, net_data, sim_len, demand, scale=None):
        self.conn = conn
        self.constants = constants
        self.v_data = None
        self.vehicles_created = 0
        self.net_data = net_data
        ###for generating vehicles
        self.origins = self.net_data['origins']
        self.destinations = self.net_data['destinations'] 
        self.lane_lengths = { l:conn.lane.getLength(l) for l in conn.lane.getIDList() }
        self.lanes = [ l for l in self.net_data['lane'] ]
        self.add_origin_routes()
        self.scale = scale
        ###stats variacles
        self.lane_vehicles = None
        self.outflow = []
        self.start_times = {}
        self.stopped_times = {}
        self.v_edge_time = {}
        self.travel_times = {}
        self.t = 0

        ###determine what function we run every step to 
        ###generate vehicles into sim
        if demand == 'single':
            self.gen_vehicles = self.gen_single
        elif demand == 'dynamic':
            self.v_schedule = self.gen_dynamic_demand(sim_len)
            self.gen_vehicles = self.gen_dynamic

    def run(self):
        lane_vehicles = self.update()
        self.gen_vehicles()
        self.t += 1
        return lane_vehicles

    def gen_dynamic(self):
        ###get next set of edges from v schedule, use them to add new vehicles
        ###this is batch vehicle generation
        try:
            new_veh_edges = next(self.v_schedule)
            self.gen_veh( new_veh_edges  )
        except StopIteration:
            print('no vehicles left')

    def gen_dynamic_demand(self, sim_len):
        ###use sine wave as rate parameter for dynamic traffic demand
        t = np.linspace(1*np.pi, 2*np.pi, sim_len)                                          
        sine = np.sin(t)+1.55
        ###create schedule for number of vehicles to be generated each second in sim
        v_schedule = []
        second = 1.0
        for t in range(int(sim_len)):
            n_veh = 0.0
            while second > 0.0:
                headway = np.random.exponential( sine[t], size=1)
                second -= headway
                if second > 0.0:
                    n_veh += 1
            second += 1.0
            v_schedule.append(int(n_veh))
                                                                                            
        ###randomly shift traffic pattern as a form of data augmentation
        v_schedule = np.array(v_schedule)
        #random_shift = np.random.randint(0,sim_len)
        random_shift = 0
        v_schedule = np.concatenate((v_schedule[random_shift:], v_schedule[:random_shift]))
        ###zero out the last minute for better comparisons because of random shift
        v_schedule[-60:] = 0
        ###randomly select from origins, these are where vehicles are generated
        v_schedule = [ np.random.choice(self.origins, size=int(self.scale*n_veh), replace = True) if n_veh > 0 else [] for n_veh in v_schedule  ]
        #v_schedule = [ np.random.choice(self.origins, size=n_veh, replace = True) if n_veh > 0 else [] for n_veh in v_schedule  ]
        ###fancy, just so we can call next for sequential access
        return v_schedule.__iter__() 

    def update(self):
        ###when a vehicle enters the network, subscribe to it for stats/data batch
        for veh_id in self.conn.simulation.getDepartedIDList():
            self.conn.vehicle.subscribe(veh_id, self.constants )

        ###use code below for sumo v <= 0.32.0
        #self.v_data = self.conn.vehicle.getSubscriptionResults()
        ###use below if sumo v >= 0.32.0
        ###subscription/batch sumo sim access is faster than API calls
        self.v_data = self.conn.vehicle.getAllSubscriptionResults()
        lane_vehicles = self.update_lane_vehicles(self.lanes)
        self.update_travel_times(self.t)
        return lane_vehicles

    def add_origin_routes(self):
        for origin in self.origins:
            self.conn.route.add(origin, [origin] )

    def gen_single(self):
        if self.conn.vehicle.getIDCount() == 0:
            ###if no vehicles in sim, spawn 1 on random link
            veh_spawn_edge = np.random.choice(self.origins)
            self.gen_veh( [veh_spawn_edge] )

    def gen_veh( self, veh_edges ):
        for e in veh_edges:
            vid = e+str(self.vehicles_created)
            self.conn.vehicle.addFull( vid, e, departLane="best" )
            self.set_veh_route(vid)
            self.vehicles_created += 1

    def set_veh_route(self, veh):
        current_edge = self.conn.vehicle.getRoute(veh)[0]
        route = [current_edge]
        while current_edge not in self.destinations:
            next_edge = np.random.choice(self.net_data['edge'][current_edge]['outgoing'])
            route.append(next_edge)
            current_edge = next_edge
        self.conn.vehicle.setRoute( veh, route )

    def update_lane_vehicles(self, lanes):
        lane_vehicles = { l:{} for l in lanes}
        for v in self.v_data:
            ###update stopped time
            if v not in self.stopped_times:
                self.stopped_times[v] = 0
            elif self.v_data[v][traci.constants.VAR_SPEED] < 0.3:
                ###increment stopped time if stopped
                self.stopped_times[v] += 1

            ###keep track for delay
            v_lane = self.v_data[v][traci.constants.VAR_LANE_ID] 
            if v_lane in self.net_data['lane']:            
                v_edge = self.net_data['lane'][v_lane]['edge'] 

                if v not in self.v_edge_time:
                   self.v_edge_time[v] = {'edge': v_edge, 't':0}
                else: 
                    if v_edge == self.v_edge_time[v]['edge']:
                        self.v_edge_time[v]['t'] += 1
                    else:
                        ###reset when new vehicle goes to new edge
                        
                        self.v_edge_time[v]['t'] = 0
                        self.v_edge_time[v]['edge'] = v_edge

            lid = self.v_data[v][traci.constants.VAR_LANE_ID]
            if lid  in lane_vehicles:
                lane_vehicles[lid][v] = self.v_data[v]
            else:
                lane_vehicles[lid] = {v:self.v_data[v]}

        self.lane_vehicles = lane_vehicles

        return lane_vehicles

    def get_edge_delay(self, edge):
        free_flow_t = self.net_data['edge'][edge]['length']/self.net_data['edge'][edge]['speed']
        edge_delay = 0

        for lane in self.net_data['edge'][edge]['lanes']:
            for v in self.lane_vehicles[lane]:
                delay =  self.v_edge_time[v]['t'] - free_flow_t
                if delay > 0:
                    edge_delay += delay
                    
        return edge_delay

    def update_travel_times(self, t):
        outflow = 0
        ####insert new vehicles
        for v in self.conn.simulation.getDepartedIDList():
            self.start_times[v] = t
            outflow += 1

        ###calculate travel time of vehicles leaving
        for v in self.conn.simulation.getArrivedIDList():
            self.travel_times[v] = t - self.start_times[v]
            del self.start_times[v]

        self.outflow.append(outflow)
