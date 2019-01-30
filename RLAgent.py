import numpy as np

class RLAgent():
    def __init__(self, neural_network, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma):
        ###this is a dict, keys = 'online', 'target'
        self.network = neural_network
        self.epsilon = epsilon
        self.exp_replay = exp_replay
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.n_exp_replay = n_exp_replay
        self.gamma = gamma
        self.experience_trajectory = []

    def get_action(self, state):
        q_state = self.network['online'].forward(state)
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            ###act randomly
            action = np.random.randint(self.n_actions)
        else:
            ###act greedily
            action = np.argmax(q_state)
        ###return action integer
        return action

    def store_experience(self, state, action, next_state, reward, terminal):
        ### here we append to a temporary experience sequence/trajectory buffer, and when terminal or steps length, at to experience replay
        experience = {'s':state, 'a':action, 'next_s':next_state, 'r':reward, 'terminal':terminal}
        self.experience_trajectory.append(experience)

        ###check if need to add trajectory to exp replay
        if len(self.experience_trajectory) == self.n_steps or terminal == True:
            self.exp_replay.append(self.experience_trajectory)
            self.experience_trajectory = []

    def train_batch(self, max_r):
        ###sample from replay
        sample_batch = self.sample_replay()
        ###process nstep, generate n step returns
        batch_inputs, batch_targets = self.process_batch(sample_batch, max_r)
        self.network['online'].backward(batch_inputs, batch_targets)

    def process_batch(self, sample_batch, max_r):
        ###each list in the sample batch is an experience trajectory
        ###use experiences in trajectory to generate targets
        processed_exps = []
        for trajectory in sample_batch:
            states, actions, rewards, next_states, terminals = [], [], [], [], []
            for exp in trajectory:
                states.append(exp['s'])
                actions.append(exp['a'])
                ###normalize reward by comparison to maximum reward 
                ###agent has experienced across all actors
                rewards.append(exp['r']/max_r)
                next_states.append(exp['next_s'])
                terminals.append(exp['terminal'])

            q_s = self.network['online'].forward(np.stack(states))

            p_exps = self.process_trajectory( states, actions, rewards, next_states, terminals, q_s )
            ###add processed experiences from trajectory to batch
            processed_exps.extend(p_exps)

        batch_inputs = np.squeeze(np.stack([ e['s'] for e in processed_exps]))
        batch_targets =  np.stack([ e['target'] for e in processed_exps])
        return batch_inputs, batch_targets

    def process_trajectory(self, states, actions, rewards, next_states, terminals, q_s ):

        if terminals[-1] is True:
            R = 0
        else:
            ###bootstrap in necessary
            q_next_s = self.network['target'].forward(next_states[-1][np.newaxis,...])
            R = np.amax(q_next_s)

        targets = self.compute_targets(rewards, R)

        exps = []
        for i in range(len(actions)):
            q_s[i, actions[i]] = targets[i]
            exps.append({'target':q_s[i], 's':states[i]})
        return exps

    def compute_targets(self, rewards, R):
        ###compute targets using discounted rewards
        target_batch = []

        for i in reversed(range(len(rewards))):
            R = rewards[i] + (self.gamma * R)
            target_batch.append(R)

        target_batch.reverse()
        return np.array(target_batch)

    def sample_replay(self):
        ###randomly sampled trajectories from shared experience replay
        idx = np.random.randint(0, self.n_exp_replay, size = self.n_batch)
        return [ self.exp_replay[i] for i in idx ]

    def set_params(self, ptype, params):
        self.network[ptype].set_weights(params)

    def get_params(self, ptype):
        return self.network[ptype].get_weights()

