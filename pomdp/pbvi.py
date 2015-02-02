import numpy
from pomdp import POMDP

class PBVI(POMDP):
    def __init__(self, states, actions, observations,
                 observation_function,
                 transition_function,
                 reward_function,
                 belief_points, 
                 gamma):
        POMDP.__init__(self, states, actions, observations,
                       observation_function,
                       transition_function,
                       reward_function, 
                       gamma)
        self.belief_points = belief_points
        self.t = 0
        self.compute_gamma_reward()

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
                    gamma_intermediate[a][o] = self.compute_gamma_action_obs(a, o)
        
            # Now compute the cross sum
            gamma_action_belief = {}
            for a in self.actions:
                gamma_action_belief[a] = {}
                for bidx in range(len(self.belief_points)):
                    b = self.belief_points[bidx]
                    gamma_action_belief[a][bidx] = self.gamma_reward[a].copy()
                    
                
                    for o in self.observations:
                        best_alpha = None
                        best_value = 0.0
                        for alpha in gamma_intermediate[a][o]:
                            val = numpy.dot(alpha, b)
                            if best_alpha is None or val > best_value:
                                best_alpha = alpha
                                best_value = val
                        gamma_action_belief[a][bidx] += best_alpha

            # Finally compute the new alpha vector set
            self.alpha_vecs = []
            for bidx in range(len(self.belief_points)):
                b = self.belief_points[bidx]
                best_alpha = None
                for a in self.actions:
                    val = numpy.dot(gamma_action_belief[a][bidx], b)
                    if best_alpha is None or val > max_val:
                        max_val = val
                        best_alpha = gamma_action_belief[a][bidx].copy()
                self.alpha_vecs.append(best_alpha)
        self.t = T

                
