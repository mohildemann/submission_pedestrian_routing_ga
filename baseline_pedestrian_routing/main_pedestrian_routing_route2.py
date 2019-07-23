import baseline_pedestrian_routing.utils as uls
from baseline_pedestrian_routing.problems import network, problem
from baseline_pedestrian_routing.algorithms.genetic_algorithm import GeneticAlgorithm
import pickle

def main_route_2( search_space_definition,pop_size, initialization = False):
    if search_space_definition == "distance_only":
        search_space = uls.preprocess_excel(r'D:\Master_Shareverzeichnis\2.Semester\Routing\Project/possible_edges_studyarea.xlsx')
    elif search_space_definition == "combined_score":
        search_space =  uls.preprocess_excel_w_scores(r'D:\Master_Shareverzeichnis\2.Semester\Routing\Project/global_local_scores_50_percent.xlsx')
    elif search_space_definition == "local_score":
        search_space = uls.preprocess_excel_w_scores(r'D:\Master_Shareverzeichnis\2.Semester\Routing\Project/local_scores_50_percent.xlsx')
    else:
        search_space = uls.preprocess_excel_w_scores(r'D:\Master_Shareverzeichnis\2.Semester\Routing\Project/global_scores_50_percent.xlsx')

    # open dumped random routes for testing (faster).
    # If the random inituialization shall be done, the section above needs to be uncommented and the init_population needs to be changed to 'routes'-
    # the pickle file was created in genetic_algorithm.py line 26

    if initialization == False:
        with open(r'..\initial_population\pop_648_3211_100.pkl', 'rb') as file_object:
            raw_data = file_object.read()
            deserialized = pickle.loads(raw_data)
            print()
        if len(deserialized)%2 == 0:
            init_population = deserialized
        else:
            init_population =  deserialized[0:-1]
    else:
        init_population = None

    problem_instance = network.Network(search_space=search_space, fitness_function=uls.sum_node_distances,start_node_id = 648, end_node_id= 3211, minimization=True)
    #this problame instance needs to be used for combination of distance and global score.
    #problem_instance = network.Network(search_space=search_space_global_score, fitness_function=uls.sum_node_distances,minimization=True)

    # setup Genetic Algorithm
    #This will then produce as many random routes until the population size is reached, if init_population was set to None
    population_size = pop_size
    pc = 0.4
    pm = 0.6
    mut_len = 9
    tournsel = 0.4
    n_iterations = 30

    for seed in range(0, 1):
        # setup random state
        random_state = uls.get_random_state(seed)
        # execute Genetic Algorithm
        #if existing population on disc shall be used, change init_population accordingly to loaded pickle
        ga1 = GeneticAlgorithm(problem_instance=problem_instance, random_state=random_state,init_population = init_population,
                               population_size=population_size, selection=uls.parametrized_tournament_selection(tournsel),
                               crossover=uls.one_point_crossover, p_c=pc,
                               mutation=uls.parametrized_iterative_bit_flip(mut_len,pm), p_m=pm)
        ga1.initialize()
        ga1.search(n_iterations = n_iterations, report = True, log = True, dplot = None)
