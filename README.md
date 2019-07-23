# submission_ga
# 1.  Downloading the files
Short description: 
## 1.1.  Code  
The code for running the Genetic Algorithm is contained in baseline_pedestrian_routing.
## 1.2.    Network
The network connections with the different costs (distance only,local_scores, global_scores, combined_scores) are contained in the folder input_edges in excel format.
## 1.3.    Precalculated initial population
If the user selects the option, that the initial population of the Genetic Algorithm shall not be randomly created but an in priori computed population shall be loaded, the data is contained in the folder initial_population in pickle format.

# 2.  Installing the requirements
In shell:
*   change directory to the directory where requirements.txt is located
*   create a virtual environment
*   activate the virtualenv
*   for installing the required packages run:  pip install -r requirements.txt in the shell
 # 3. The logic of the code
 ## 3.1 Define the searchspace and problem
 The searchspace are all possible solutions. In this case, it is the matrix with all nodes with the distances to the neighbor nodes. This is contained in the excel files in the folder input_edges. The problem is a combinatorical and is basically solving the routing problem. The problem class type is a network. The problem has defined a fitness function and states if it is a minimization or maximization problem. In this case it is a minimization problem, as the total distance shall be as small as possible.
 ## 3.2 Defining the parameters
Parameters in main_pedestrian_routing: 
first parameter is search_space_definition. Can be "distance_only", "local_score", "combined_score" or "global_score"
second parameter is the population size
third parameter is the choice, if a random initialization for the initital population shall be done or if one will be used that was saved as a pickle file and loaded. Default value is False, as it is much faster. Set it to True, if the random initialization shall be executed.
These parameters are only for the convenience of the user.

Parameters in route specific main (like in main_pedestrian_route1):
population_size = size of the total population
pc = possibility of a crossover
pm = mutation probability
mut_len = path length of the sub-path with the mutated nodes
tournsel = selection pressure of the tournament selection
n_iterations = number of iterations

This literature describes in high detail what the parameters of the route specific main (like in main_pedestrian_route1) do:
Sivanandam, S. N.; Deepa, S. N. (2007): Introduction to genetic algorithms. Berlin, New York: Springer.

 ## 3.3 Searching the best routes
 After the initialization of the Genetic Algorithm the search for the best possible route begins. The main code for this is in genetic_algorithm.py, which calls most of the necessary functions in utils. First of all, a logger is set up to save the necessary information of the search like parameters, the fitness values of the solutions and the node ids of the best inidividual.
 1. Selection: The used method is tournament selection
 2. Crossover: The used method is a n point crossover
 3. Mutation: The used method is a sub path mutation
 4. Replacement: Duplicates and invalid solutions are replaced by random solutions
 5. Elitism: The best solution of each generation is saved.
 
 # 4 Results
 Results are stored in LogFIles
 
