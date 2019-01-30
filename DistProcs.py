from Proc import ActorProc, LearnerProc
from NetworkData import NetworkData
from SumoSim import SumoSim
import math
from multiprocessing import *
import numpy as np
from help_funcs import save_data, load_data

class DistProcs():
    def __init__(self, n_actors, n_learners, mode, net_fp, args):
        self.args = args
        if mode == 'train':
            ###at least one learner
            if n_actors < 1:
                n_learners += n_actors - 1
                n_actors = 1
            ###when training, create different exploration rates for actors
            ###line space from exclusive explortation (1.0) to greedy (0.0)
            actor_eps = np.linspace(1.0, args.eps, num = n_actors)
        elif mode == 'test':
            if n_learners < 1:
                n_actors += n_learners - 1
                n_learners = 1
            ###when testing, all actors have same exploration rate
            actor_eps = [args.eps for _ in range(n_actors)]

        ##NetworkData is a dict of information about elements of the simulation network (e.g., lanes, edges, intersections)
        if args.load == True:
            ###read saved dictionary
            self.net_data = load_data('net_data.p')
        else:
            ##read some network file (*.xml)
            net_data = NetworkData(net_fp).get_net_data()
            ##start sumo sim once and get traffic light/traffic signal control (tsc) data using API calls
            sim = SumoSim(args.port, 0, args.sumo_cfg )
            nogui = True
            sim.gen_sim(nogui, args.sim_len)
                                                                                                          
            tsc_data = sim.get_tsc_data()
            sim.cleanup()
                                                                                                          
            ###include tsc data in net data
            net_data['tsc'] = tsc_data
            self.net_data = net_data

        self.agents = [tsc for tsc in self.net_data['tsc']]
        ###distribute agents across learner procs
        learner_agents = self.assign_learner_agents(self.agents, n_learners)

        ###use to synchronize procs
        barrier = Barrier(n_actors+n_learners)

        ###use this mp shared dict for data between procs
        manager = Manager()
        self.rl_stats = manager.dict({})
        for i in self.net_data['tsc']:
            self.rl_stats[i] = manager.dict({})
            self.rl_stats[i]['n_exp'] = 0
            self.rl_stats[i]['updates'] = 0
            self.rl_stats[i]['max_r'] = 1.0
            self.rl_stats[i]['online'] = None
            self.rl_stats[i]['target'] = None
            self.rl_stats['n_sims'] = 0
            self.rl_stats['total_sims'] = 104
            self.rl_stats['delay'] = manager.list()
            self.rl_stats['queue'] = manager.list()
            self.rl_stats['throughput'] = manager.list()

        #exp_replay = manager.dict({ tsc: manager.list() for tsc in self.net_data['tsc']  })
        ###create shared memory for experience replay (governs agents appending and learners accessing and deleting)
        exp_replay = manager.dict({ tsc: manager.list() for tsc in self.agents })

        ###create list of desired proces, allocate agents evenly across learners
        actor_procs = [ActorProc(a, args, exp_replay, self.rl_stats, self.net_data, actor_eps[a], barrier) for a in range(n_actors)]
        learner_procs = [LearnerProc(l, args, exp_replay, self.rl_stats, self.net_data, learner_agents[l], barrier ) for l in range(n_learners)]

        self.procs = actor_procs + learner_procs

        #self.write_hp(args)

    def run(self):
        print('...Starting DistProcs')
        ###start everything   
        for p in self.procs:
            p.start()
                              
        ###join when finished
        for p in self.procs:
            p.join()

        if self.args.save is True:
            self.save_agent_weights()
        print('...finishing DistProcs')
    
    def assign_learner_agents(self, agents, n_learners):
        learner_agents = [ [] for _ in range(n_learners)]
        i = 0
        for agent in agents:
            learner_agents[i%n_learners].append(agent)
        ##list of lists, each sublit is the agents a learner is responsible for
        return learner_agents

    def save_agent_weights(self):
        agent_weights = {}
        for agent in self.agents:
            agent_weights[agent] = self.rl_stats[agent]['online']
        
        save_data('saved_weights.p', agent_weights)
        save_data('net_data.p', self.net_data)
        print('... finished saving weights')

    def write_hp(self, args):
        with open('hpresults.csv', 'a') as f:
            hp = [args.lr, args.lre, args.replay, args.updates, args.batch, args.target]
            f.write('/')
            for h in hp:
                f.write(str(h)+',')
            f.write('\n' )
