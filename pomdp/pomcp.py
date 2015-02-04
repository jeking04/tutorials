
node_count = 0

class Node:
    def __init__(self, h, parent_id=None, V_init=0, N_init=0):
        self.h = h
        self.V = V_init
        self.N = N_init
        self.B = []
        self.id = node_count
        self.parent = parent_id
        node_count++

class Tree:
    def __init__(self):
        self.nodes = {}
        
    def contains(self, h):
        for nid, node in nodes.iteritems():
            if node.h == h:
                return True
            
        return False

    def add(self, h, parent_id):
        n = Node(h, parent_id=parent_id)
        self.nodes[n.id] = n
        return n

    def get_node(self, h):
        for nid, node in nodes.iteritems():
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
                 rollout_policy,
                 simulator,
                 c = 0.5):
           

        POMDP.__init__(self, states, actions, observations,
                       observation_function,
                       transition_function,
                       reward_function, 
                       gamma)

        self.tree = Tree()
        self.rollout_policy = rollout_policy
        self.simulator = simulator
        self.c = c

    def rollout(self, state, h, depth, max_depth):
        if depth > max_depth:
            return 0

        # Select an action, generate an observation and recurse rollout
        #  until the maximum depth has been achieved
        a = self.rollout_policy(h)
        s_new, o, r = self.simulator(state, a)
        return r + self.gamma * self.rollout(s_new, h + [o,a], depth+1)

    def simulate(self, state, h, parent, depth, max_depth):

        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0

        # Initialize child nodes and return an approximate reward for this
        #  history by rolling out until max depth
        if parent is None:
            n = self.tree.add(h, parent.id)
            for a in self.actions:
                self.tree.add( h + [a], n.id )
            return self.rollout(state, h, depth, max_depth)

        # Find the action that maximizes 
        node_h = self.tree.get_node(h)
        values = numpy.zeros(len(self.actions))
        for idx in range(len(self.actions))
            n = self.tree.get_node(h + [self.actions[idx]])
            values[idx] = n.V + self.c * numpy.sqrt(numpy.log(node_h.N)/n.N)
        aidx = numpy.argmax(values)
        a = self.actions[aidx]
    
        # Perform monte-carlo simulation of the state under the action
        s_new, o, r = self.simulator(state, a)
        
        # Calculate the reward
        node_ha = self.tree.get_node(h + [a])
        R = r + self.gamma * self.simulate(s_new, h + [ao], node_ha,
                                           depth+1, max_depth)

        # Update the node for h
        node_h.B += [s]
        node_h.N += 1
        self.tree.update_node(node_h)

        # Update the child node for this action
        node_ha.N += 1
        node_ha.V += (R - node_ha.V)/node_ha.N
        self.tree.update_node(node_ha)

        return R

    def solve(self, T):
        '''
        Solves for up to T steps
        '''
        s = self.sample_state()
        self.simulate(s, [], 0, T)
