##got tf 1.12 optimized packages from https://github.com/lakshayg/tensorflow-build, installed with Anaconda Python 3.6.6

###force CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop

class NeuralNetwork():
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre):
        ###create FF neural network (i.e., Q-function approximator)
        self.model = Sequential()
        self.model.add(Dense(hidden_d[0], input_shape=(input_d,), activation=hidden_act))
        if len(hidden_d) > 0:
            for h in hidden_d[1:]:                              
                self.model.add(Dense(h, activation=hidden_act))
        self.model.add(Dense( output_d, activation=output_act))
        ###compile model with loss and optimizer
        #adam = Adam(lr=lr, epsilon=lre )
        adam = RMSprop(lr=lr, epsilon=lre )
        self.model.compile(optimizer=adam, loss='mse')

    def forward(self, _input):
        return self.model.predict(_input)

    def backward(self, _input, _target):
        self.model.fit(_input, _target, batch_size = 1, epochs = 1,  verbose=0 )

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)
        
if __name__ == '__main__':
    ###test neural network
    input_d = 10
    output_d = 4
    batch_size = 16
    nn = NeuralNetwork(input_d, [20,20], 'relu', output_d, 'linear', 0.001, 0.000001 )
    x = np.random.randn(batch_size, input_d) 
    y = np.random.randn(batch_size, output_d) 
    nn.backward(x, y)
