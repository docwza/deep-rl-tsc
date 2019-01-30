from multiprocessing import *
from NeuralNetwork import NeuralNetwork
from RLAgent import RLAgent
from SumoSim import SumoSim
from help_funcs import load_data
import time

def gen_neural_networks( agents, net_data, hact, oact, lr, lre):
    print('...started creating neural networks for agents')
    networks = {}
    for i in agents:
        networks[i] = {}
        #input_d = self.get_input_d( self.args.nn, self.args.s_hist, self.netdatadict['i'][i])
        ###1 neuron for each lane density, and one hot for action and terminal
        input_d = len(net_data['tsc'][i]['inc_lanes']) + net_data['tsc'][i]['n_green_phases'] + 1
        ###2 hidden layer of ns*|s| neurons
        ns = 2
        hidden_d = [input_d*ns, input_d*ns]
        output_d = net_data['tsc'][i]['n_green_phases']
        networks[i]['online'] = NeuralNetwork( input_d, hidden_d,  hact, output_d, oact, lr , lre)
        networks[i]['target'] = NeuralNetwork( input_d, hidden_d,  hact, output_d, oact, lr , lre)

    print('...finished creating neural networks for agents')
    return networks

class ActorProc(Process):
    def __init__(self, idx, args, exp_replay, rl_stats, net_data, eps, barrier ):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.exp_replay = exp_replay
        self.rl_stats = rl_stats
        self.net_data = net_data
        self.eps = eps
        self.barrier = barrier
        ###everything is indexed by agent name

    def run(self):
        ###create neural networks for all tsc agents in simulation 
        print('...starting actor '+str(self.idx))
        agents = [tsc for tsc in self.net_data['tsc']]
        print(agents)
        print('ACTOR neural networks')
        print('ACTOR neural networks')
        agent_networks = gen_neural_networks(agents, self.net_data, self.args.hact, self.args.oact, self.args.lr, self.args.lre)

        self.barrier.wait()
        ###load weights from learners
        for tsc in agent_networks:
            agent_networks[tsc]['online'].set_weights( self.rl_stats[tsc]['online'] )

        port = self.args.port + self.idx
        sim = SumoSim(port, self.idx, self.args.sumo_cfg)

        if self.args.mode == 'test':
            print('---- Running test sims ----')
            for _ in range(1):
                sim.gen_sim(self.args.nogui, self.args.sim_len)                                             
                ###run the sim until completion, pass in neural network here                                        
                sim.run(self.net_data, self.args, self.exp_replay, agent_networks, self.eps, self.rl_stats)         
            self.barrier.wait()

        elif self.args.mode == 'train':
            print('---- Running train sims ----')
            ###fill exp replays of all agents
            while not self.replays_full(agents):
                sim.gen_sim(self.args.nogui, self.args.sim_len)                                             
                ###run the sim until completion, pass in neural network here
                sim.run(self.net_data, self.args, self.exp_replay, agent_networks, self.eps, self.rl_stats)

            print('------- finished filling exp replays, start learning ---------')
            self.barrier.wait()

            ###keep acting until sufficient updates from learners
            while not self.finished_acting(agents):
                sim.gen_sim(self.args.nogui, self.args.sim_len)                                             
                ###run the sim until completion, pass in neural network here                                        
                sim.run(self.net_data, self.args, self.exp_replay, agent_networks, self.eps, self.rl_stats)         

        print('...end actor '+str(self.idx))

    def replays_full(self, agents):                                  
        ###only start learning once replay buffer is full               
        for agent in agents:                                            
            if self.rl_stats[agent]['n_exp'] < self.args.replay:        
                return False                                            
        return True                                                     
                                                                     
    def finished_acting(self, agents):                                  
        for agent in agents:                                            
            if self.rl_stats[agent]['updates'] < self.args.updates:     
                return False                                            
        return True                                                     

class LearnerProc(Process):
    def __init__(self, idx, args, exp_replay, rl_stats, net_data, agent_ids, barrier ):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.exp_replay = exp_replay
        self.rl_stats = rl_stats
        self.net_data = net_data
        self.agent_ids = agent_ids
        self.barrier = barrier

    def run(self):
        ####loop thru agents
        print('LEARNER neural networks')
        agent_networks = gen_neural_networks([tsc for tsc in self.agent_ids], self.net_data, self.args.hact, self.args.oact, self.args.lr, self.args.lre)

        ###load weights if we want
        if self.args.load is True:
            agent_weights = load_data('saved_weights.p')
            for tsc in self.agent_ids:
                agent_networks[tsc]['online'].set_weights( agent_weights[tsc] )

        ###send weights to actors
        for tsc in self.agent_ids:
            weights = agent_networks[tsc]['online'].get_weights()
            ###send to actors
            self.rl_stats[tsc]['online'] = weights
            agent_networks[tsc]['target'].set_weights( weights )
        ###ensure all learners have sent weights before starting
        self.barrier.wait()

        ###create rl agents using neural networks
        rl_agents = { tsc:RLAgent(agent_networks[tsc], self.args.eps, self.exp_replay[tsc], self.net_data['tsc'][tsc]['n_green_phases'], self.args.n_steps, self.args.batch, self.args.replay, self.args.gamma) for tsc in self.agent_ids}
        
        ###wait until sufficient exp in replay to start making updates
        self.barrier.wait()

        ###timer for stats
        self.last_update = time.time()
        period = 60

        self.learn_time = time.time()

        if self.args.mode == 'train':
            ###reset n_exp count
            for agent in self.agent_ids:
                self.rl_stats[agent]['n_exp'] = 0

            while not self.finished_learning():                                                
                for tsc in rl_agents:                                                         
                    ###only do batch updates after something has been added to exp replay
                    if self.rl_stats[tsc]['n_exp'] > 0:
                        rl_agents[tsc].train_batch( self.rl_stats[tsc]['max_r'])
                        self.rl_stats[tsc]['n_exp'] -= 1
                        self.rl_stats[tsc]['updates'] += 1
                        ###send online weight to shared dict for actors
                        self.rl_stats[tsc]['online'] = rl_agents[tsc].get_params('online')
                                                                                               
                        ###clip exp replay
                        diff = len(self.exp_replay[tsc]) - self.args.replay
                        if diff > 0:
                            del self.exp_replay[tsc][:diff]
                                                                                               
                        ###update target network to online on regular interval
                        if self.rl_stats[tsc]['updates'] % self.args.target == 0:
                            ###set target to online params
                            rl_agents[tsc].set_params('target', self.rl_stats[tsc]['online'])
                ###try stats
                t = time.time() - self.last_update
                if t > period:
                    print('========= AGENT EXP PROGRESS UPDATE LEARNER '+str(self.idx)+' =====')
                    self.print_stats()
                    self.last_update = time.time()
                    T = time.time()-self.learn_time
                    ###use min progress agent as the ETA estimate p
                    min_progress = min([ self.rl_stats[agent]['updates']/float(self.args.updates) for agent in self.agent_ids])
                    eta = self.ETA( min_progress, T )
                    print('==== ETA seconds: '+str( round(eta, 0) )+' minutes: '+str( round(eta/60.0, 2) )+' hours: '+str( round(eta/3600.0, 2) )+' ====')
                    #print(str(ETA( np.amin([ self.rl_stats[agent]['updates']/float(self.args.updates for agent in self.agent_ids])), T))
        print('...end learner '+str(self.idx))


    def ETA(self, p, t):
        if p == 0.0:
            return 0.0
        else:
            return ((1.0/p)*t)*(1.0-p)

    def print_stats(self):
        agent_progress = []
        for agent in self.agent_ids:
            agent_progress.append( self.rl_stats[agent]['updates']/float(self.args.updates) )
        print(str(self.agent_ids)+'\n'+str(agent_progress))

    def finished_learning(self):
        for agent in self.agent_ids:
            if self.rl_stats[agent]['updates'] < self.args.updates:
                return False
        return True
