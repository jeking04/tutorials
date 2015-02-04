import random, numpy
from pomdp import POMDP

class Node:
    def __init__(self, nid, h, parent_id=None, V_init=0, N_init=0):
        self.h = h
        self.V = V_init
        self.N = N_init
        self.B = []
        self.id = nid
        self.parent = parent_id


class Tree:
    def __init__(self):
        self.nodes = {}
        
    def contains(self, h):
        for nid, node in self.nodes.iteritems():
            if node.h == h:
                return True
            
        return False

    def add(self, h, parent):
        nid = len(self.nodes.keys())
        if parent is not None:
            n = Node(nid, h, parent_id=parent.id)
        else:
            n = Node(nid, h)
        self.nodes[n.id] = n
        return n

    def get_node(self, h):
        for nid, node in self.nodes.iteritems():
            if node.h == h:
                return node
        return None

    def update_node(self, n):
        self.nodes[n.id] = n

class POMCP(POMDP):
    
    def __init__(self, states, actions, observations,
                 observation_function,
                 transition_function,
                 reward_function,
                 gamma,
                 initial_belief,
                 c = 0.5):
           

        POMDP.__init__(self, states, actions, observations,
                       observation_function,
                       transition_function,
                       reward_function, 
                       gamma)

        self.tree = Tree()
        self.initial_belief = initial_belief
        self.c = c
        self.a_selected = None

    def simulate_action(self, state, action):
        # pick a random number between 0 and 1
        r = random.random()
        total = 0.0
        for idx in range(len(self.states)):
            total += self.transition_function(self.states[idx], action, state)
            if r < total:
                s_new = self.states[idx]
                break

        r = random.random()
        total = 0.0
        for idx in range(len(self.observations)):
            total += self.observation_function(self.observations[idx], s_new)
            if r < total:
                o_new = self.observations[idx]
        
        r = self.reward_function(state, action)

        return s_new, o_new, r

    def random_rollout(self, h):
        aidx = random.randint(0, len(self.actions)-1)
        return self.actions[aidx]

    def rollout(self, state, h, depth, max_depth):
        if depth > max_depth:
            return 0

        # Select an action, generate an observation and recurse rollout
        #  until the maximum depth has been achieved
        a = self.random_rollout(h)
        s_new, o, r = self.simulate_action(state, a)
        return r + self.gamma * self.rollout(s_new, h + [o,a], 
                                             depth+1, max_depth)

    def simulate(self, state, h, parent, depth, max_depth):
        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0

        # Initialize child nodes and return an approximate reward for this
        #  history by rolling out until max depth
        node_h = self.tree.get_node(h)
        if node_h is None:
            n = self.tree.add(h, parent)
            for a in self.actions:
                self.tree.add( h + [a], n )
            return self.rollout(state, h, depth, max_depth)

        # Find the action that maximizes 
        values = numpy.zeros(len(self.actions))
        for idx in range(len(self.actions)):
            n = self.tree.get_node(h + [self.actions[idx]])
            values[idx] = n.V
            if n.N > 0 and node_h.N > 0:
                values[idx] += self.c * numpy.sqrt(numpy.log(node_h.N)/n.N)
        aidx = numpy.argmax(values)
        a = self.actions[aidx]
    
        # Perform monte-carlo simulation of the state under the action
        s_new, o, r = self.simulate_action(state, a)
        
        # Calculate the reward
        node_ha = self.tree.get_node(h + [a])
        R = r + self.gamma * self.simulate(s_new, h + [a,o], node_ha,
                                           depth+1, max_depth)

        # Update the node for h
        node_h.B += [state]
        node_h.N += 1
        self.tree.update_node(node_h)

        # Update the child node for this action
        node_ha.N += 1
        node_ha.V += (R - node_ha.V)/node_ha.N
        self.tree.update_node(node_ha)

        return R

    def sample_state(self, b):
        r = random.random()
        idx = sum(r >= numpy.cumsum(b))
        return self.states[idx]
    
    def solve(self, T):
        '''
        Solves for up to T steps
        '''
        for i in range(10):
            s = self.sample_state(self.initial_belief)
            self.simulate(s, [], None, 0, T)

        vals = numpy.zeros(len(self.actions))
        for aidx in range(len(self.actions)):
            a = self.actions[aidx]
            vals[aidx] = self.tree.get_node([a]).V

        self.a_selected = self.actions[numpy.argmax(vals)]
        return self.a_selected

    def draw(self, beliefs):
        pass

    def get_action(self, belief):
        print 'Returning action: ',self.a_selected
        return self.a_selected
