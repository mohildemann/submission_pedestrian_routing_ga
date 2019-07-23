import logging
import numpy as np
from functools import reduce
from baseline_pedestrian_routing.algorithms.random_search import RandomSearch
from baseline_pedestrian_routing.solutions.solution import Solution
import pickle
from time import gmtime, strftime

class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size, init_population,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m
        self.init_population = init_population

    def initialize(self):
        if self.init_population is not None:
            self.population = self.init_population[:self.population_size]

        else:
            self.population = self._generate_random_valid_chromosomes()
            self.init_population = self.population
            routes_dump = pickle.dumps(self.population)
            with open(r'D:\Master_Shareverzeichnis\2.Semester\Routing\Project\initial_population\pop_1815_3211_100.pkl','wb') as file_object:
                file_object.write(routes_dump)
        self.best_solution = self._get_elite(self.population)
        print("best solution of initial population: " + str(self.best_solution.fitness))

    def search(self, n_iterations, report=False, log=False, dplot=None, remove_duplicates = True):
        if log:
            t = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
            lgr = logging.getLogger(t)
            lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
            # add a file handler
            fh = logging.FileHandler(r'LogFiles/'+t+'.csv')
            fh.setLevel(logging.DEBUG)
            frmt = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
            fh.setFormatter(frmt)
            lgr.addHandler(fh)
            log_event = [self.problem_instance.__class__, id(self._random_state), __name__,self.population_size, self.selection.__name__, self.crossover.__name__, self.p_c,
                             self.mutation.__name__, self.p_m,]
            lgr.info(','.join(list(map(str, log_event))))

        if dplot is not None:
            dplot.background_plot(self.problem_instance.search_space, self.problem_instance.fitness_function)

            def _iterative_plot():
                points = np.array([chromosome.representation for chromosome in self.population])
                points = np.insert(points, points.shape[0], values=self.best_solution.representation, axis=0)
                points = np.vstack((points[:, 0], points[:, 1]))
                z = np.array([chromosome.fitness for chromosome in self.population])
                z = np.insert(z, z.shape[0], values=self.best_solution.fitness)
                dplot.iterative_plot(points, z, self.best_solution.fitness)

        for iteration in range(n_iterations):
            offsprings = []

            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state) for _ in range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                if not (hasattr(off1, 'fitness') and hasattr(off2, 'fitness')):
                    self.problem_instance.evaluate(off1)
                    self.problem_instance.evaluate(off2)
                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            elite_offspring = self._get_elite(offsprings)
            self.best_solution = self._get_best(self.best_solution, elite_offspring)

            if report:
                self._verbose_reporter_inner(self.best_solution, iteration)

            if log:
                log_event = [iteration, self.best_solution.fitness,self._phenotypic_diversity_shift(offsprings), self.best_solution.representation[:,0], [sol.fitness for sol in self.population]]
                lgr.info(','.join(list(map(str, log_event))))

            # replacement
            if self.best_solution == elite_offspring:
                self.population = offsprings
            else:
                self.population = offsprings
                #if the individual does not already exist in the population replace a random one with the best of the old generation
                if self.best_solution not in self.population:
                    index = self._random_state.randint(self.population_size)
                    self.population[index] = self.best_solution

            #the following block removes inidividuals with the same fitness values and replaces them with random individuals in order to keep diversity
            if remove_duplicates is not None:
                pop_without_duplicates = np.array([]).reshape(0,1)
                i = 0
                for solution in self.population:
                    if not np.isin(solution.fitness, pop_without_duplicates).any():
                        pop_without_duplicates = np.vstack([pop_without_duplicates,solution.fitness])
                    else:
                        random_solution_id = self._random_state.randint(0, self.init_population.shape[0])
                        new_rand_solution = self.init_population[random_solution_id]
                        self.population[i] = new_rand_solution
                    i = i + 1

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, chromosome):
        mutant = self.mutation(chromosome.representation, self._random_state, self.problem_instance.search_space)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _phenotypic_diversity_shift(self, offsprings):
        fitness_parents = np.array([parent.fitness for parent in self.population])
        fitness_offsprings = np.array([offspring.fitness for offspring in offsprings])
        return np.std(fitness_offsprings)-np.std(fitness_parents)

    def _generate_random_valid_chromosomes(self):
        chromosomes = np.array([self._generate_random_valid_solution()
                              for _ in range(self.population_size)])
        return chromosomes