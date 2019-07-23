import numpy as np
from baseline_pedestrian_routing.problems.problem import Problem
from baseline_pedestrian_routing.solutions._item import sum_weights
from baseline_pedestrian_routing.solutions.solution import Solution
from baseline_pedestrian_routing.utils import random_route
import baseline_pedestrian_routing.utils as uls

class Network(Problem):
    def __init__(self, search_space, fitness_function,start_node_id, end_node_id, minimization=False):
        Problem.__init__(self, search_space, fitness_function, minimization)
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id

    def evaluate(self, solution):
        nodes = solution.representation
        #the following code repairs/eliminates repeating nodes and edges
        nodes = uls.repair_route(nodes)
        solution.valid = self._validate(nodes)

        if solution.valid:
            solution.fitness = self.fitness_function(nodes)
        else:
            if self.minimization:
                solution.fitness = np.iinfo(np.int32).max
            else:
                solution.fitness = 0


    def _validate(self, nodes):
        valid = True
        node_checks = []
        for i in range(nodes.shape[0]-1):

            node_check = False
            for j in range(nodes[i+1][1][:, 0].shape[0]):
                if int(nodes[i+1][1][j][0])==nodes[i][0]:
                    node_check = True
                else:
                    g = nodes[i]
                    h = nodes[i+1]
            node_checks.append([i,node_check])
            if node_check is False:
                valid = False
        if valid is not False:
            valid = True
        return valid



    def sample_search_space(self, random_state):
        return Solution(random_route(self.search_space, random_state, self.start_node_id, self.end_node_id))