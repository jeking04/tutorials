from pomdp import POMDP
import copy, numpy

class Exact(POMDP):
    
    def __init__(self, states, actions, observations,
                 observation_function,
                 transition_function,
                 reward_function,
                 gamma,
                 pruning_beliefs):

        POMDP.__init__(self, states, actions, observations,
                       observation_function,
                       transition_function,
                       reward_function, 
                       gamma)
        self.compute_gamma_reward()
        self.t = 0
        self.pruning_beliefs = pruning_beliefs

    def solve(self, T):
        '''
        Perform value iteration for up to T steps
        '''
        for idx in range(self.t, T):
            print 'Computing update for time: ', idx
            
            # First compute a set of updated vectors for every action/observation pair
            gamma_intermediate = {}
            for a in self.actions:
                gamma_intermediate[a] = {}
                for o in self.observations:
                    gamma_intermediate[a][o] = self.compute_gamma_action_obs(a,o)

            # Now compute the cross sum
            gamma_all = {}
            self.alpha_vecs = []
            for a in self.actions:
                gamma_in = [self.gamma_reward[a].copy()]
                    
                for obs in gamma_intermediate[a]:
                    gamma_out = []
                    for alpha_int in gamma_intermediate[a][obs]:
                        for gamma in gamma_in:
                            gamma_out.append(gamma + alpha_int.copy())
                    gamma_in = copy.copy(gamma_out)
                for gamma in gamma_in:
                    self.alpha_vecs.append(gamma)
            self.alpha_vecs = numpy.array(self.alpha_vecs)
            self.prune()

        self.t = T

    def prune(self):

        all_values = numpy.dot(self.alpha_vecs, self.pruning_beliefs.transpose())
        chosen_alpha = numpy.unique(numpy.argmax(all_values, axis=0))
        self.alpha_vecs = self.alpha_vecs[chosen_alpha,:]
