import argparse, os, itertools, time
import numpy as np
from DistProcs import DistProcs

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nogui", default=False, action='store_true', dest='nogui', help='disable gui, default: False')
    parser.add_argument("-actor", type=int, default=os.cpu_count()-1, dest='n_actors', help='number of actor procs (parallel simulations) generating experiences, default: os.cpu_count()-1')
    parser.add_argument("-learner", type=int, default=1, dest='n_learners', help='number of learner procs (parallel simulations) producing updates, default: 1')

    parser.add_argument("-sumocfg", type=str, default='networks/double.sumocfg', dest='sumo_cfg', help='path to desired simulation configuration file, default: networks/double.sumocfg' )
    parser.add_argument("-port", type=int, default=9000, dest='port', help='sumo port, default: 9000' )
    parser.add_argument("-simlen", type=int, default=7200, dest='sim_len', help='length of simulation in seconds/steps')
    parser.add_argument("-netfp", type=str, default='networks/double.net.xml', dest='net_fp', help='path to desired simulation network file, default: networks/double.net.xml')
    parser.add_argument("-scale", type=float, default=1.0, dest='scale', help='automatic flow scaling parameter (i.e., >1.0, more vehicles, default: 1.0')


    parser.add_argument("-mode", type=str, default='train', dest='mode', help='reinforcement mode, train (agents receive updates) or test (no updates), default:train'  )

    ##rl paramns
    parser.add_argument("-shist", type=int, default=1, dest='s_hist')
    parser.add_argument("-lr", type=float, default=0.0001, dest='lr', help='neural network learning rate, default: 0.0001')
    #parser.add_argument("-nn", type=str, default='conv', dest='nn')
    parser.add_argument("-eps", type=float, default=0.05, dest='eps', help='q-learning exploration rate, default: 0.05'  )
    parser.add_argument("-gamma", type=float, default=0.99, dest='gamma', help='reinforcement learning discount factor, default: 0.99')
    parser.add_argument("-lre", type=float, default=0.00000001, dest='lre', help='neural network learning rate epsilon, default: 0.00000001')

    parser.add_argument("-tsc", type=str, default='rl', dest='tsc', help='traffic signal control type, reinforcement learning (rl), cyclic fixed (fixed), uniform cycle fixed (uniform), default: rl')

    parser.add_argument("-replay", type=int, default=10000, dest='replay', help='maximum number of experience in replay, default: 10 000')
    parser.add_argument("-batch", type=int, default=16, dest='batch', help='number of samples in training batch, default: 16')
    parser.add_argument("-target", type=int, default=50, dest='target', help='target network update period relative to batch updates, default: 50')
    parser.add_argument("-updates", type=int, default=10000, dest='updates', help='total number of batch updates during training, default: 10 000' )
    parser.add_argument("-oact", type=str, default='linear', dest='oact', help='neural network output layer activation function (tanh, sigmoid, linear), default: linear')
    parser.add_argument("-hact", type=str, default='relu', dest='hact', help='neural network hidden layer activation function (tanh, sigmoid, linear), default: relu')

    parser.add_argument("-load", default=False, action='store_true', dest='load', help='load saved weights and net data from training, default: False')
    parser.add_argument("-save", default=False, action='store_true', dest='save', help='save network weights and net data after training for future use, default: False')

    ###n step q learning
    parser.add_argument("-nsteps", type=int, default=1, dest='n_steps', help='n-steps in experience trajectory for updates, default: 1')
    parser.add_argument("-arepeat", type=int, default=15, dest='a_repeat', help='action repeat length (length of actions/green phases in seconds/steps), default: 15' )

    ##sim params
    parser.add_argument("-vlen", type=float, default=7.5, dest='v_len', help='average vehicle length+headway for density state calculation, default: 7.5')
    parser.add_argument("-demand", type=str, default='dynamic', dest='demand', help='vehicle demand (dynamic, single) defined in Vehicles.py, default: dynamic')

    ###minimum tsc params
    parser.add_argument("-red", type=int, default=4, dest='yellow_t', help='length of yellow change phases in seconds/steps, default: 4')
    parser.add_argument("-yellow", type=int, default=4, dest='red_t', help='length of red clearance phases in seconds/steps, default: 4')
    parser.add_argument("-green", type=int, default=15, dest='green_t')
    #parser.add_argument("-ming", type=int, default=10, dest='ming')
    #parser.add_argument("-maxg", type=int, default=45, dest='maxg')

    args = parser.parse_args()
    return args

def main():
    start_time = time.time()

    args = parse_cl_args()

    dist_procs = DistProcs( args.n_actors, args.n_learners, args.mode, args.net_fp, args)
    dist_procs.run()
    print('Total program run time: '+str((time.time()-start_time)/60.0)+' minutes')


if __name__ == '__main__':
    main()
