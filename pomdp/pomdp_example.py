#!/usr/bin/env python
from pbvi import PBVI
from exact import Exact
from pomcp import POMCP
import argparse, numpy

states = ['s0', 's1', 's2']
controls = ['a0', 'a1', 'a2']
measurements = ['o0', 'o1']

R = [[-100., 100., -1.], # x1
     [ 100., -50., -1.], # x2
     [ 0., 0., 0.]] # x3
def reward_function(x, u):
    xidx = states.index(x)
    uidx = controls.index(u)
    return R[xidx][uidx]


Tr = [[[0, 0, 1],[0, 0, 1], [0, 0, 1]], # u1
     [[0, 0, 1],[0, 0, 1], [0, 0, 1]], # u2
     [[0.2, 0.8, 0.], [0.8, 0.2, 0.], [0., 0., 1.]]] #u3

def transition_function(x_new, u, x_orig):
    x_new_idx = states.index(x_new)
    x_orig_idx = states.index(x_orig)
    uidx = controls.index(u)
    return Tr[uidx][x_orig_idx][x_new_idx]

O = [[0.7, 0.3, 0.5], # z1
     [0.3, 0.7, 0.5]] # z2

def observation_function(z, x):
    zidx = measurements.index(z)
    xidx = states.index(x)
    return O[zidx][xidx]

def generate_belief(stepsize):
    
    beliefs = []
    for p in numpy.arange(0., 1.+stepsize, stepsize):
        beliefs.append([p, 1.-p, 0.])

    return numpy.array(beliefs)

def run(T, algo, stepsize, gamma):
    beliefs = generate_belief(stepsize)
    print 'Number of belief points: ', len(beliefs)

    initial_belief = [0.99, 0.01, 0.0]
    
    if algo == 'pbvi':
        pomdp = PBVI(states, controls, measurements, 
                     observation_function, 
                     transition_function,
                     reward_function,
                     beliefs,
                     gamma)
    elif algo == 'exact':
        pomdp = Exact(states, controls, measurements,
                      observation_function,
                      transition_function,
                      reward_function,
                      gamma,
                      beliefs)
    elif algo == 'pomcp':
        pomdp = POMCP(states, controls, measurements,
                      observation_function,
                      transition_function,
                      reward_function,
                      gamma, 
                      initial_belief)

    pomdp.solve(T)
    pomdp.get_action(initial_belief)
    pomdp.draw(beliefs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Solve pomdp')
    parser.add_argument('time_horizon', type=int, help='The time horizon for the solver')
    parser.add_argument('--gamma', type=float, default=1.0, help='The discount factor to apply')
    parser.add_argument('--stepsize', type=float, default=0.01, help='For PBVI, the stepsize between belief points')
    parser.add_argument('--algo', type=str, default='pbvi', choices=['pbvi', 'exact', 'pomcp'],
                        help='The algorithm for the solver to use')
    args = parser.parse_args()

    run(args.time_horizon, args.algo, args.stepsize, args.gamma)
