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
        self.experience_rollout = []

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
        ### here we append to a temporary experience sequence/rollout buffer, and when terminal or steps length, at to experience replay
        experience = {'s':state, 'a':action, 'next_s':next_state, 'r':reward, 'terminal':terminal}
        self.experience_rollout.append(experience)

        ###check if need to add rollout to exp replay
        if len(self.experience_rollout) == self.n_steps or terminal == True:
            self.exp_replay.append(self.experience_rollout)
            self.experience_rollout = []

    def train_batch(self, max_r):
        ###sample from replay
        sample_batch = self.sample_replay()
        ###process nstep, generate n step returns
        batch_inputs, batch_targets = self.process_batch(sample_batch, max_r)
        self.network['online'].backward(batch_inputs, batch_targets)

    def process_batch(self, sample_batch, max_r):
        ###sample batch should be a list of lists
        ###each list is an experience rollout
        ###use experiences to generate targets
        processed_exps = []
        for rollout in sample_batch:
            states, actions, rewards, next_states, terminals = [], [], [], [], []
            for exp in rollout:
                states.append(exp['s'])
                actions.append(exp['a'])
                rewards.append(exp['r']/max_r)
                next_states.append(exp['next_s'])
                terminals.append(exp['terminal'])

            q_s = self.network['online'].forward(np.stack(states))

            p_exps = self.process_rollout( states, actions, rewards, next_states, terminals, q_s )
            ###add processed experiences from rollout to batch
            processed_exps.extend(p_exps)

        batch_inputs = np.squeeze(np.stack([ e['s'] for e in processed_exps]))
        batch_targets =  np.stack([ e['target'] for e in processed_exps])
        return batch_inputs, batch_targets

    def process_rollout(self, states, actions, rewards, next_states, terminals, q_s ):

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
        size = len(rewards)
        y_batch = []

        for i in reversed(range(size)):
            R = rewards[i] + (self.gamma * R)
            y_batch.append(R)

        y_batch.reverse()
        return np.array(y_batch)

    def sample_replay(self):
        ###sample this funny way because of mp shared exp replay list, might not be necessary
        idx = np.random.randint(0, self.n_exp_replay, size = self.n_batch)
        return [ self.exp_replay[i] for i in idx ]

    def set_params(self, ptype, params):
        self.network[ptype].set_weights(params)

    def get_params(self, ptype):
        return self.network[ptype].get_weights()

