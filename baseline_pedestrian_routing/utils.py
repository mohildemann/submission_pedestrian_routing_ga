import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

def get_random_state(seed):
    return np.random.RandomState(seed)

def search_random_neighbor_node(total_nodes,node, random_state):
    #this block returns a random neighbor node to the input node
    poss_neighbors = []
    for i in total_nodes:
        for j in i[1]:
            if j[0]==node[0]:
                poss_neighbors.append(i)
    if len(poss_neighbors) > 1:
        random_index = random_state.randint(0, len(poss_neighbors))
    else:
        random_index = 0
    if len(poss_neighbors)>=1:
        return poss_neighbors[random_index]
    else:
        return False

def random_route(searchspace, random_state, start_node_id, end_node_id):
    #this block searches for a random route from the start_node to the end_node
    node_beginning = searchspace[np.where(searchspace[:, 0] == start_node_id)[0][0]]
    node_end = searchspace[np.where(searchspace[:, 0] == end_node_id)[0][0]]
    possible_route_beginning = []
    possible_route_end = []
    possible_route_beginning.append(node_beginning)
    possible_route_end.append(node_end)
    i=0
    df_node_exist = pd.DataFrame(columns=['node_id', 'route_beginning_exists', 'route_ending_exists', 'node_type_beginning','node_type_ending','valid_meetpoint'])
    df_node_exist['node_id'] = searchspace[:, 0]
    df_node_exist=df_node_exist.assign(route_beginning_exists=False,route_ending_exists=False,valid_meetpoint=False)
    df_node_exist.set_index('node_id', inplace=True)
    #prepare check, if one of the nodes can connect to some of the existing points
    #all nodes are stored in a pandas df. If
    while i <50000 and not True in df_node_exist['valid_meetpoint'].unique():
        try:
            possible_node1 = search_random_neighbor_node(searchspace,possible_route_beginning[-1], random_state)
            possible_node2 = search_random_neighbor_node(searchspace, possible_route_end[-1], random_state)
            i = i +1
            if possible_node1 is not False:
                possible_route_beginning.append(possible_node1)
                df_node_exist.at[possible_node1[0],'route_beginning_exists'] = True
                if df_node_exist.at[possible_node1[0],'route_ending_exists'] == True:
                    df_node_exist[possible_node1[0]]['valid_meetpoint'] = True
                    meeting_node_id = possible_node1[0]
                df_node_exist.at[possible_node1[0], 'node_type_beginning'] = 'mainnode'
                #if nested list has more element than 1, it needs to be flattened
                # Do not do it for last node
                if np.shape(possible_node1[1])[0]>1 and possible_node1[0] != node_end[0]:
                    for j in range(len(possible_node1[1][:, 0].flatten())):
                        df_node_exist.at[possible_node1[1][j][0], 'route_beginning_exists'] = True
                        df_node_exist.at[possible_node1[1][j][0], 'node_type_beginning'] = possible_node1[0]
                        if df_node_exist.at[int(possible_node1[1][j][0]),'route_ending_exists'] == True:
                            df_node_exist.at[possible_node1[1][j][0],'valid_meetpoint'] = True
                            meeting_node_id = possible_node1[1][j][0]
                # if nested list has only one element
                else:
                    df_node_exist.at[possible_node1[1][0][0], 'route_beginning_exists'] = True
                    df_node_exist.at[possible_node1[1][0][0], 'node_type_beginning'] = possible_node1[0]
                    if df_node_exist.at[int(possible_node1[1][0][0]), 'route_ending_exists'] == True:
                        df_node_exist.at[possible_node1[1][0][0],'valid_meetpoint'] = True
                        meeting_node_id = possible_node1[1][0][0]
            #Same approach, only from the other side of the route (starts at endpoint)
            if possible_node2 is not False:
                possible_route_end.append(possible_node2)
                df_node_exist.at[possible_node2[0], 'route_ending_exists'] = True
                if df_node_exist.at[possible_node2[0],'route_beginning_exists'] == True:
                    df_node_exist[possible_node2[0]]['valid_meetpoint'] = True
                    meeting_node_id = possible_node2[0]
                df_node_exist.at[possible_node2[0], 'node_type_ending'] = 'mainnode'
                #if node has many neighbours, add them to list. Do not do it for last node
                if np.shape(possible_node2[1])[0] > 1 and possible_node2[0] != node_end[0]:
                    for j in range(len(possible_node2[1][:, 0].flatten())):
                        df_node_exist.at[possible_node2[1][j][0], 'route_ending_exists'] = True
                        df_node_exist.at[possible_node2[1][j][0], 'node_type_ending'] = possible_node2[0]
                        if df_node_exist.at[int(possible_node2[1][j][0]),'route_beginning_exists'] == True:
                            df_node_exist.at[possible_node2[1][j][0],'valid_meetpoint'] = True
                            meeting_node_id = possible_node2[1][j][0]
                else:
                    df_node_exist.at[possible_node2[1][0][0], 'route_ending_exists'] = True
                    df_node_exist.at[possible_node2[1][0][0], 'node_type_ending'] = possible_node2[0]
                    if df_node_exist.at[int(possible_node2[1][0][0]), 'route_beginning_exists'] == True:
                        df_node_exist.at[possible_node2[1][0][0],'valid_meetpoint'] = True
                        meeting_node_id = possible_node2[1][0][0]
        except:
            continue
    last_node_from_beginning = possible_route_beginning[-1]
    last_node_from_end = possible_route_end[-1]
    if True in df_node_exist['valid_meetpoint'].unique():
        #print(meeting_node_id)
        meeting_node_info = df_node_exist.loc[meeting_node_id,:]
        if meeting_node_info['node_type_beginning'] == 'mainnode':
            cutting_point_id_route_beginning,node_type_beginning = meeting_node_id, 'mainnode'
        else:
            cutting_point_id_route_beginning,node_type_beginning = meeting_node_info['node_type_beginning'], 'neighbor'
        if meeting_node_info['node_type_ending'] == 'mainnode':
            cutting_point_id_route_ending,node_type_ending = meeting_node_id, 'mainnode'
        else:
            cutting_point_id_route_ending,node_type_ending = meeting_node_info['node_type_ending'],'neighbor'
        # if both connecting nodes are only in the neighbor column, the common neighbor node needs to be delivered
        necessary_connecting_neighbor = None
        if node_type_ending != 'mainnode' and node_type_beginning != 'mainnode':
            boolArr = (searchspace[:, 0] == meeting_node_id)
            result = np.where(boolArr)
            necessary_connecting_neighbor = searchspace[result[0][0]]
        # get positions in boths routes, delete the unnessecary elements
        combined_random_route = combine_routes(possible_route_beginning,possible_route_end,cutting_point_id_route_beginning,cutting_point_id_route_ending, connecting_neighbor = necessary_connecting_neighbor)
        resulting_route = repair_route(combined_random_route)
    else:
        print('No valid meeting point found. Returns empty array')
        resulting_route = None
    return resulting_route


def combine_routes(in_route_1, in_route_2, cuttingpoint_route1,cuttingpoint_route2, connecting_neighbor = None):
    possible_route_beginning = np.array(in_route_1)
    possible_route_end = np.array(in_route_2)
    boolArr = (possible_route_beginning[:, 0] == cuttingpoint_route1)
    result = np.where(boolArr)
    boolArr2 = (possible_route_end[:, 0] == cuttingpoint_route2)
    result2 = np.where(boolArr2)
    possible_route_beginning = possible_route_beginning[0:result[0][0] + 1]
    possible_route_end = possible_route_end[0:result2[0][0] + 1]
    # combine np arrays, if invert is True (default) --> 2nd array flipped (reversed order, as the endpoint is the beginning point of possible_round_end)
    # Also, if connecting_neighbor is necessay (not of type none, see declaration, it is stacked inbetween)
    if connecting_neighbor is None:
        combined_route = np.vstack([possible_route_beginning, np.flip(possible_route_end,0)])
    else:
        combined_route = np.vstack([possible_route_beginning,connecting_neighbor, np.flip(possible_route_end,0)])
    return combined_route

def combine_mutated_route(in_route,  mutated_node, reenter_node , mutated_segments):
    ## Route --> niedrigerer Schnittpunktindex: cut --> mutationroute einfügen --> cut sobald zweiter Schnittpunktindex erreicht ist --> zweiter Teil Route
    boolArr = (in_route[:, 0] == mutated_node)
    result = np.where(boolArr)
    first_part_route = in_route[0:result[0][0]]
    boolArr2 = (in_route[:, 0] == reenter_node[0])
    result2 = np.where(boolArr2)
    last_part_route = in_route[result2[0][0]:]
    boolArr3 = (mutated_segments[:, 0] == mutated_node)
    result3 = np.where(boolArr3)
    boolArr4 = (mutated_segments[:, 0] == reenter_node[0])
    result4 = np.where(boolArr4)
    # if first cuttingpoint of in_route is not the first one in mutated segments, flip and find the new index
    if result3[0][0] > result4[0][0]:
        mutated_segments = np.flip(mutated_segments, 0)
        boolArr3 = (mutated_segments[:, 0] == mutated_node)
        result3 = np.where(boolArr3)
        boolArr4 = (mutated_segments[:, 0] == reenter_node[0])
        result4 = np.where(boolArr4)
    middle_part_route = mutated_segments[result3[0][0]:result4[0][0]]
    combined_mutated_route = np.vstack([first_part_route, middle_part_route, last_part_route])
    combined_mutated_repaired_route = repair_route(combined_mutated_route)
    return combined_mutated_repaired_route

def repair_route(unfixed_route):
    for i in unfixed_route[:-1,0]:
        boolArr = (unfixed_route[:, 0] == i)
        result = np.where(boolArr)
        if result[0].size > 1:
            # print("route error found")
            # print('i: '+str(i))
            # print('min_position: ' + str(result[0].min()))
            # print('max_position: ' + str(result[0].max()))
            unfixed_route = np.delete(unfixed_route, np.s_[result[0].min():result[0].max()],axis = 0)
    return  unfixed_route

def random_boolean_1D_array(length, random_state):
    return random_state.choice([True, False], length)


def parametrized_iterative_bit_flip(mutation_path_length,  prob):
    def bit_flip(route, random_state,search_space ):
        route_before_mutation = route
        if random_state.uniform() < prob:
            for i in route:
                i=0
                index = random_state.randint(2, route.shape[0]-2)
                while index == route[0][0] or index == route[-1][0]:
                    index = random_state.randint(0, route.shape[0])
                connection_exists = False
                mutation_path = np.array([]).reshape(0,2)
                #get random neighbors of the random node
                for i in range(mutation_path_length):
                    if i == 0:
                        possible_node1 = search_random_neighbor_node(search_space, route[index], random_state)
                        mutation_path=np.vstack([mutation_path, route[index]])
                        mutation_path = np.vstack([mutation_path, possible_node1])
                    else:
                        possible_node1 = search_random_neighbor_node(search_space, mutation_path[-1], random_state)
                        if possible_node1[0] != mutation_path[-2][0]:
                            mutation_path = np.vstack([mutation_path,  possible_node1])
                #check if random point is a possible connection node to the input route. If it is not a direct neighbor in the input route, delete the elements inbetween
                possible_connection_points = np.array([]).reshape(0,4)
                for mutation_node in mutation_path[1:]:
                    boolArr = (mutation_node[0]==route[:, 0])
                    result = np.where(boolArr)
                    for i in result[0]:
                        #check, if not already a direct neighbor node
                        if  np.isin(mutation_node[0], [ route[index-1][0], route[index][0],route[index+1][0]]).any() == False:
                            #structure: selected node for mutation, reentry node of route, Necessary connecting node, mutation path
                            possible_connection_points = np.vstack([possible_connection_points, [route[index][0], mutation_node , False ,mutation_path]])

                        else:
                            pass
                    #also check possible connection with the neighbors of the input route

                    for neigbhbornode in mutation_node[1][:, 0].flatten():
                        #select the possible reenter node of the route
                        boolArr = (neigbhbornode == route[:, 0])
                        result = np.where(boolArr)
                        if (result[0].size) >= 1 and np.isin(neigbhbornode, [ route[index-1][0], route[index][0],route[index+1][0]]).any() == False:
                            # get the full node of the connection_neighbor
                            boolArr = (search_space[:, 0] == neigbhbornode)
                            result = np.where(boolArr)
                            valid_neigbhbornode = search_space[result[0][0]]
                            # we have the random mutation node, the reenter node and the needed neighbor node. The neighbor node needs to be added in the middle.
                            boolArr = (mutation_node[0] == mutation_path[:, 0])
                            result = np.where(boolArr)
                            #if several possibilities exist for reconnecting, choose random one. The +1 only says, that
                            if (result[0].size) > 1:
                                conn_point = np.random.choice(result[0], 1)
                                mutation_path = np.insert(mutation_path, conn_point + 1, valid_neigbhbornode, axis=0)
                            else:
                                mutation_path = np.insert( mutation_path, result[0][0]+1, valid_neigbhbornode, axis=0)
                            # structure: selected node for mutation, reentry node of route, Necessary connecting node, mutation path
                            possible_connection_points = np.vstack([possible_connection_points, [route[index][0], valid_neigbhbornode, mutation_node,mutation_path]])
                #if connecting neighbor is necessary, append to list

                #only mutate if id is not start or endpoint
                boolArr = ( route[0][0] == possible_connection_points[:, 0])
                boolArr2 =  (route[-1][0] == possible_connection_points[:, 0])
                result = np.where(boolArr)
                result2 = np.where(boolArr2)
                if result[0].size < 1 and result[0].size < 1:
                    if possible_connection_points.shape[0]>1:
                        random_possible_connection_point = possible_connection_points[random_state.randint(0, possible_connection_points.shape[0])]
                        comb_route = combine_mutated_route(route, mutated_node=random_possible_connection_point[0],
                                                           reenter_node=random_possible_connection_point[1],
                                                           mutated_segments=random_possible_connection_point[3])
                        route = comb_route
                    elif possible_connection_points.shape[0]==1:
                        random_possible_connection_point = possible_connection_points[0]
                        comb_route = combine_mutated_route(route, mutated_node = random_possible_connection_point[0], reenter_node = random_possible_connection_point[1],
                                     mutated_segments = random_possible_connection_point[3])
                        route = comb_route
                ###Suche haut hin, jetzt müssen nur noch ausgangsroute mit dem mutation_path verknüpft werden:
                ## Route --> niedrigerer Schnittpunktindex: cut --> mutationroute einfügen --> cut sobald zweiter Schnittpunktindex erreicht ist --> zweiter Teil Route
        return route
    return bit_flip

def random_float_1D_array(hypercube, random_state):
    return np.array([random_state.uniform(tuple_[0], tuple_[1])
                     for tuple_ in hypercube])


def random_float_cbound_1D_array(dimensions, l_cbound, u_cbound, random_state):
    return random_state.uniform(lower=l_cbound, upper=u_cbound, size=dimensions)


def parametrized_ball_mutation(radius):
    def ball_mutation(point, random_state):
        return np.array(
            [random_state.uniform(low=coordinate - radius, high=coordinate + radius) for coordinate in point])

    return ball_mutation

def sum_node_distances(nodes):
    sum_distance = 0
    for i in range(nodes.shape[0]-1):
        try:
            node = nodes[i]
            next_node = nodes[i+1]
            g = np.where(np.isin(next_node[0], node[1][:, 0]))
            l_index = g[0][0]
            sum_distance = sum_distance + (node[1][l_index, 1])
        except:
            sum_distance = np.iinfo(np.int32).max
    return float(sum_distance)

def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=0)#len(point.shape) % 2 - 1)


def rastrigin(point):
    a = len(point) * 10 if len(point.shape) <= 2 else point.shape[0] * 10
    return a + np.sum(point ** 2 - 10. * np.cos(np.pi * 2. * point), axis=0)


def one_point_crossover(p1_r, p2_r, random_state):
    possible_crossover_points = np.array([]).reshape(0,3)
    #First find all possible connection points
    for i in range(p1_r.shape[0]):
        #check if direct neighbor
        if np.isin(p1_r[i][0],p2_r[:,0]).any():
            possible_crossover_points = np.vstack([possible_crossover_points, [p1_r[i],'direct', None]])
        #if not direct neighbor, check if neighbor is indirect
        else:
            for j in range(p1_r[i][1].shape[0]):
                if np.isin(p1_r[i][1][j][0] ,p2_r[:,0]).any():
                    possible_crossover_points = np.vstack([possible_crossover_points, [p1_r[i],'neighbor', p1_r[i][1][j][0]]])
    len_ = possible_crossover_points.shape[0]
    #point is the random crossover point. randint(1,len_-1) excludes the first and last node in calc of the random number, as they are always in the other array
    point = random_state.randint(1,len_-1)
    if possible_crossover_points[point][1]=='direct':
        #look for position of random point in parent 1
        boolArr = (p1_r[:,0] == possible_crossover_points[point][0][0])
        result = np.where(boolArr)
        # look for position of random point in parent 2
        boolArr2 = (p2_r[:, 0] == possible_crossover_points[point][0][0])
        result2 = np.where(boolArr2)
        #offspring one gets first part of parent 1 and second part of parent 2
        off1_r = np.vstack([p1_r[:result[0][0]], p2_r[result2[0][0]:]])
        # offspring two gets opposite elements
        off2_r = np.vstack([p2_r[:result2[0][0]], p1_r[result[0][0]:]])

    #if the neighbor is not direct, the connection point needs to be included
    elif possible_crossover_points[point][1]=='neighbor':
        # look for position of random point in parent 1
        boolArr = (p1_r[:, 0] == possible_crossover_points[point][0][0])
        result = np.where(boolArr)
        boolArr2 = (p2_r[:, 0] == possible_crossover_points[point][0][0])
        result2 = np.where(boolArr2)
        # if the first element of result is longer than 0, then the first parent contains the connecting neighbor in the first column.
        # if both contain the neighbor only in the second column (neighbornodes), the results are both of length 0

        #in this case, parent 1 contains connecting point in first column with node_id and parent 2 does not
        if len(result[0])>0 and len(result2[0])<1:
            l1 = len(result2[0])
            l2 = len(result[0])
            #look again for node in first column of parent 1
            boolArr = (p1_r[:, 0] == possible_crossover_points[point][0][0])
            result = np.where(boolArr)
            # look for position of random point in parent 2, but in second column
            boolArr2 = (p2_r[:, 0] == possible_crossover_points[point][2])
            result2 = np.where(boolArr2)
            #offspring one gets first part of parent 1 and second part of parent 2
            off1_r = np.vstack([p1_r[0:result[0][0]+1], p2_r[result2[0][0]:]])
            # offspring two gets opposite elements
            off2_r = np.vstack([p2_r[0:result2[0][0]+1], p1_r[result[0][0]:]])

        # in this case, it is the opposite
        elif len(result[0]<1) and len(result2[0]>0):
            l1 = len(result2[0])
            l2 = len(result[0])
            # look again for node in first column of parent 1
            boolArr = (p1_r[:, 0] == possible_crossover_points[point][2])
            result = np.where(boolArr)
            # look for position of random point in parent 2, but in second column
            boolArr2 = (p2_r[:, 0] == possible_crossover_points[point][0][0])
            result2 = np.where(boolArr2)
            # offspring one gets first part of parent 1 and second part of parent 2
            off1_r = np.vstack([p1_r[0:result[0][0]+1], p2_r[result2[0][0]:]])
            # offspring two gets opposite elements
            off2_r = np.vstack([p2_r[0:result2[0][0]+1], p1_r[result[0][0]:]])
        #in this case, the connecting node is in the second column with the neighbors

        else:
            l1 = len(result2[0])
            l2 = len(result[0])
            # look again for node in first column of parent 1
            boolArr = (p1_r[:, 0] == possible_crossover_points[point][2])
            result = np.where(boolArr)
            # look for position of random point in parent 2, but in second column
            boolArr2 = (p2_r[:, 0] == possible_crossover_points[point][2])
            result2 = np.where(boolArr2)
            # offspring one gets first part of parent 1 and second part of parent 2
            off1_r = np.vstack([p1_r[0:result[0][0]+1], p2_r[result2[0][0]:]])
            # offspring two gets opposite elements
            off2_r = np.vstack([p2_r[0:result2[0][0]+1], p1_r[result[0][0]:]])


    return off1_r, off2_r


def generate_cbound_hypervolume(dimensions, l_cbound, u_cbound):
    return [(l_cbound, u_cbound) for _ in range(dimensions)]


def parametrized_ann(ann_i):
    def ann_ff(weights):
        return ann_i.stimulate(weights)

    return ann_ff


def parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population) * pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        if minimization:
            return reduce(lambda x, y: x if x.fitness <= y.fitness else y, tournament_pool)
        else:
            return reduce(lambda x, y: x if x.fitness >= y.fitness else y, tournament_pool)

    return tournament_selection

def preprocess_excel(xls_file):
    df = pd.read_excel(xls_file)  # for an earlier version of Excel, you may need to use the file extension of 'xls'
    df = df.sort_values(['u'], ascending=[1])
    problem_space = np.array([]).reshape(0,2)
    for i in range(df.shape[0]):
        if i >=1:
            #if u or v already exists in dataframe add to nested list
            if np.isin(df['u'][i],problem_space[:, 0]).any():
                #add entry for existing node
                g=np.where(problem_space[:,0] == df['u'][i])
                l_index = g[0][0]
                list_n = [df['v'][i],df['length_m'][i]]
                problem_space[l_index][1]=np.vstack([problem_space[l_index][1], list_n])
            else:
                # new entry for the nodes if  u node does not exist yet
                problem_space = np.vstack([problem_space, [df['u'][i], [[df['v'][i], df['length_m'][i]]]]])
            # add entry for other way round (adding neighbor entry in both directions: if node 1 is neighbor of node 2, node 2 is also the neighbor of node 1)
            if np.isin(df['v'][i],problem_space[:, 0]).any():
                g = np.where(problem_space[:, 0] == df['v'][i])
                l_index = g[0][0]
                list_n = [df['u'][i], df['length_m'][i]]
                problem_space[l_index][1] = np.vstack([problem_space[l_index][1], list_n])
            else:
                #new entry for the nodes if v node does not exist yet
                problem_space = np.vstack([problem_space, [df['v'][i], [[df['u'][i], df['length_m'][i]]]]])
        else:
            #problem_space = [[df['u'][0],[[df['v'][0],df['length_m'][0]]]]]
            problem_space = np.vstack([problem_space, [df['u'][0], [[df['v'][0], df['length_m'][0]]]]])
            problem_space = np.vstack([problem_space, [df['v'][0], [[df['u'][0], df['length_m'][0]]]]])
    #problem_space = np.sort(problem_space, order = '0')
    for i in range(problem_space.shape[0]):
        if isinstance(problem_space[i][1], list):
            problem_space[i][1] = np.array(problem_space[i][1])
    return problem_space

def preprocess_excel_w_scores(xls_file):
    df = pd.read_excel(xls_file)  # for an earlier version of Excel, you may need to use the file extension of 'xls'
    df = df.sort_values(['u'], ascending=[1])
    problem_space = np.array([]).reshape(0,2)
    for i in range(df.shape[0]):
        if i >=1:
            #if u or v already exists in dataframe, add to nested list
            if np.isin(df['u'][i],problem_space[:, 0]).any():
                #add entry for existing node
                g=np.where(problem_space[:,0] == df['u'][i])
                l_index = g[0][0]
                list_n = [df['v'][i],df['new_dist_uv'][i]]
                problem_space[l_index][1]=np.vstack([problem_space[l_index][1], list_n])
            else:
                # new entry for the nodes if  u node does not exist yet
                problem_space = np.vstack([problem_space, [df['u'][i], [[df['v'][i], df['new_dist_uv'][i]]]]])
            # add entry for other way round (adding neighbor entry in both directions: if node 1 is neighbor of node 2, node 2 is also the neighbor of node 1)
            # other than in the function preprocess_excel, the two distances can differ
            if np.isin(df['v'][i],problem_space[:, 0]).any():
                g = np.where(problem_space[:, 0] == df['v'][i])
                l_index = g[0][0]
                list_n = [df['u'][i], df['new_dist_vu'][i]]
                problem_space[l_index][1] = np.vstack([problem_space[l_index][1], list_n])
            else:
                #new entry for the nodes if v node does not exist yet
                problem_space = np.vstack([problem_space, [df['v'][i], [[df['u'][i], df['new_dist_vu'][i]]]]])
        else:
            #problem_space = [[df['u'][0],[[df['v'][0],df['length_m'][0]]]]]
            problem_space = np.vstack([problem_space, [df['u'][0], [[df['v'][0], df['new_dist_uv'][0]]]]])
            #problem_space = np.vstack([problem_space, [df['v'][0], [[df['u'][0], df['new_dist_vu'][0]]]]])
    #problem_space = np.sort(problem_space, order = '0')
    for i in range(problem_space.shape[0]):
        if isinstance(problem_space[i][1], list):
            problem_space[i][1] = np.array(problem_space[i][1])
    return problem_space


class Dplot():
    def background_plot(self, hypercube, function_):
        dim1_min = hypercube[0][0]
        dim1_max = hypercube[0][1]
        dim2_min = hypercube[1][0]
        dim2_max = hypercube[1][1]
        x0 = np.arange(dim1_min, dim1_max, 0.1)
        x1 = np.arange(dim2_min, dim2_max, 0.1)
        x0_grid, x1_grid = np.meshgrid(x0, x1)
        x = np.array([x0_grid, x1_grid])
        y_grid = function_(x)
        # plot
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlim(dim1_min, dim1_max)
        self.ax.set_ylim(dim2_min, dim2_max)
        self.ax.plot_surface(x0_grid, x1_grid, y_grid, rstride=1, cstride=1, color="green", alpha=0.15) # cmap=cm.coolwarm,

    def iterative_plot(self, points, z, best=None):
        col = "k" if best is None else np.where(z == best, 'r', 'k')
        size = 75 if best is None else np.where(z == best, 150, 75)
        self.scatter = self.ax.scatter(points[0], points[1], z, s=size, alpha=0.75, c=col)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.scatter.remove()