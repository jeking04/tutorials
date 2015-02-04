import numpy
import matplotlib.pyplot as plt

class POMDP:
    def __init__(self, states, actions, observations,
                 observation_function,
                 transition_function,
                 reward_function,
                 gamma):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.observation_function = observation_function
        self.transition_function = transition_function
        self.reward_function = reward_function
        self.gamma = gamma
        self.alpha_vecs = [numpy.zeros(len(self.states))]

    def compute_gamma_reward(self):
        '''
        Computes a set of |A| vectors each of length |S|
        representing the reward for each state/action pair
        '''
        self.gamma_reward = {}
        for a in self.actions:
            v = numpy.zeros(len(self.states))
            for idx in range(len(self.states)):
                s = self.states[idx]
                v[idx] = self.reward_function(s, a)
            self.gamma_reward[a] = v

    def compute_gamma_action_obs(self, a, o):
        '''
        Computes a set of vectors, one for each previous alpha
        vector that represents the update to that alpha vector
        given an action and observation
        '''
        
        gamma_action_obs = []
        for alpha in self.alpha_vecs:
            v = numpy.zeros(len(self.states)) # initialize the update vector
            for sidx in range(len(self.states)):
                s = self.states[sidx]
                
                for s_prime_idx in range(len(self.states)):
                    s_prime = self.states[s_prime_idx]
                    v[sidx] += self.transition_function(s, a, s_prime)*self.observation_function(o, s_prime)*alpha[s_prime_idx]
                v[sidx] *= self.gamma # discount
            gamma_action_obs.append(v)
        return gamma_action_obs

    def solve(self, T):
        pass

    def draw(self, belief_points):
        '''
        Draw each of the alpha vectors and the final solution
        '''
        vals = numpy.zeros((len(self.alpha_vecs), len(belief_points)))
        for aidx in range(len(self.alpha_vecs)):
            alpha = self.alpha_vecs[aidx]
            vals[aidx,:] = numpy.dot(alpha, belief_points.transpose())
            plt.plot(belief_points[:,0], vals[aidx,:], '--')
            plt.hold(True)
        
        max_vals = numpy.max(vals, axis=0)
        plt.plot(belief_points[:,0], max_vals, 'k', linewidth=2)
        plt.show()

    def get_action(self, belief):
        vals = numpy.zeros(len(self.actions))
        for aidx in range(len(self.actions)):
            vals[aidx] = numpy.dot(self.alpha_vecs[aidx], belief)
        print vals
        aidx = numpy.argmax(vals)
        action = self.actions[aidx]
        
        print 'Returning action: ', action
        return action
