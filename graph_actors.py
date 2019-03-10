import os
import matplotlib.pyplot as plt
import numpy as np

def get_actor_data():
    actor_csv = [ f for f in os.listdir() if '.csv' in f]

    actor_data = {}
    for csv in actor_csv:
        ###use exploration rate as the key, truncate because its a float
        explore_rate = csv.split('.csv')[0][:4]
        actor_data[explore_rate] = np.genfromtxt(csv, delimiter=',')
    return actor_data

def graph_actor_data(data):
    plt.rcParams["font.size"] = 24

    ###actors may not have all completed same number of sims
    ###graph up to the smallest
    x = np.amin([ data[d].size  for d in data ])
    y_data = np.concatenate([ data[d] for d in data ])

    data_keys = sorted([d for d in data], reverse=True)

    for d in data_keys:
        plt.plot(np.arange(x), data[d][:x], label=d, linewidth=5.0)

    plt.ylim(0, np.amax(y_data)*1.1)
    plt.xlim(0,x)

    ###titles and labels text
    plt.title('Distributed Q-learning Training Performance')
    plt.ylabel('Episode Average Travel Time (s)')
    plt.xlabel('Episodes')
    plt.legend(loc='upper right', title='Agent Exploration Rates')

    ###inset axes showing last few episodes
    L = 5
    a = plt.axes([0.4, 0.5, 0.3, 0.3])
    for d in data_keys:
        plt.plot(np.arange(x)[-L:], data[d][x-L:x], linewidth=5.0)

    plt.xlim(np.min(np.arange(x)[-L:]), np.max(np.arange(x)[-L:]))
    plt.title('Final Training Episodes Performance')

    plt.show()

def main():
    ###get data
    data = get_actor_data()
    ###graph data
    graph_actor_data(data)

if __name__ == '__main__':
    main()
