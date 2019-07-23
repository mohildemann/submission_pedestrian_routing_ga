from baseline_pedestrian_routing import main_pedestrian_routing_route1,main_pedestrian_routing_route2,main_pedestrian_routing_route3

#parameters for functions:
# first parameter is search_space_definition. Can be "distance_only", "local_score", "combined_score" or "global_score"
# second parameter is the population size
# third parameter is the choice, if a random initialization for the initital population shall be done or if one will be used that was saved as a pickle file and loaded.
# Default value is False, as it is much faster. Set it to True, if the random initialization shall be executed.

#Execution for route 1
main_pedestrian_routing_route1.main_route_1("distance_only",20, False)

#Execution for route 2
main_pedestrian_routing_route2.main_route_2("combined_score",30, False)

#Execution for route 3
main_pedestrian_routing_route3.main_route_3("local_score",50, True)
