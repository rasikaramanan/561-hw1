import random
import math 
import bisect
import warnings 
import numpy as np

def create_init_population(size, cities):
    """ 
    size = size of population to create
    cities = list of city locations parsed from input
    returns: list of paths (roundtrip -- ends with starting city) of length = size
    """

    if size > math.factorial(len(cities)):
        warnings.warn(f"Given initial population size of {size} is greater than the number of permutations possible for {len(cities)} cities. Setting initial population size to the max allowable value, {math.factorial(len(cities))}.")
        size = math.factorial(len(cities))
        
    paths = []
    orders_tried = {}
    city_indices = list(range(len(cities)))
    
    for _ in range(size):
        city_order = city_indices.copy()
        #print("in loop, city_order: ", city_order)
        while True: # ensures no duplicate paths
            random.seed()  # Reseed with a system-generated random value
            random.shuffle(city_order)
            #print("in while loop, dict contents before addition attempt: ", orders_tried)
            if tuple(city_order) not in orders_tried:
                orders_tried[tuple(city_order)] = None    
                #print("in while loop, dict contents after successful addition: ", orders_tried)
                break
            #print("in while loop, addition of following failed: ", city_order)

        #print("after while loop, city order: ", city_order)
        city_order.append(city_order[0]) # add starting city to end of list
        #print("city order after appending starting city: ", city_order)
        path = [cities[j] for j in city_order] # populate w cities in city_order
        #print("path created: ", path)
        paths.append(path)
    
    return paths
    

def calc_path_distance(path):
    path_arr = np.array(path)  
    
    if np.array_equal(path_arr[0], path_arr[-1]):
        path_arr = path_arr[:-1]
    
    shifted_path = np.roll(path_arr, shift=-1, axis=0)  # Shifted version to compute distances
    distances = np.linalg.norm(shifted_path - path_arr, axis=1)  # Vectorized Euclidean distance
    
    return np.sum(distances)

def make_rank_list(init_pop):
    rank_list = []
    for index, path in enumerate(init_pop):
        entry = (calc_path_distance(path), index)
        # will automatically sort by entry's val at index 0
        bisect.insort(rank_list, entry)
    return rank_list

def get_top_10_percent(num_pop):
    top10percent = num_pop // 10 
    if top10percent == 0: # 3 cities in init pop
        top10percent += 2
    elif top10percent % 2 == 1:
        top10percent += 1
    return top10percent

def get_roulette_wheel_percent(num_pop):
    num_mp = num_pop // 10 * 4
    if num_mp == 0:
        num_mp = 2
    return num_mp

def create_mating_pool(population, rank_list):

    # ensure the top10percent is an even num
    top10percent = get_top_10_percent(len(rank_list)) 
    
    # 40% of population to be selected for mating from roulette wheel
    mp_from_roulette_wheel = get_roulette_wheel_percent(len(rank_list))

    # store the top 10 percent of population in top10_ranked, rest stays in rank_list
    top10_ranked, rank_list = rank_list[:top10percent], rank_list[top10percent:]

    population_dict = {index: route for index, route in enumerate(population)}

    # mating_pool is a list of paths, init with top10_ranked
    mating_pool = [population_dict[index] for (_, index) in top10_ranked]

    # Invert fitness scores (lower is better, so higher probability)
    inv_fitness_rank_list = []
    sum_inv_fitness = 0.0

    for dist, index in rank_list:
        inv_fit = 1.0 / dist
        inv_fitness_rank_list.append((inv_fit, index))
        sum_inv_fitness += inv_fit
    
    parent_indices = []
    
    while len(parent_indices) < mp_from_roulette_wheel: #SELECT MATING POOL OF SIZE = 2
        rand_num = random.uniform(0, sum_inv_fitness)
        #print("random number is: ", rand_num)
        sum_pop = 0
        for score, index in inv_fitness_rank_list:
            sum_pop += score
            if rand_num < sum_pop:
                if index not in parent_indices: # ensures duplicate parents do not occur
                    parent_indices.append(index)
                    # print("PARENT ADDED: ", score, index)
                break 
    # print("parent indices selected: ", parent_indices)
    population_dict = {index: route for index, route in enumerate(population)}
    mating_pool += [population_dict[index] for index in parent_indices]
    return mating_pool

def get_rand_pairs(mating_pool):
    if len(mating_pool) % 2 == 1:
        mating_pool.pop()

    random.shuffle(mating_pool)
    return list(zip(mating_pool[::2], mating_pool[1::2]))

def crossover(parent1, parent2, start_index, end_index):
    """ 
    need to know -- indices of subarr1 containing data that is not in subarr2
    need to know -- content from subarr2 that is not in subarr1

    """
    n = len(parent1)
    child = [None] * n
    visited = {}

    for i in range(start_index, end_index + 1):
        child[i] = parent1[i]
        visited[tuple(parent1[i])] = True
    #visited now contains parent1 cities from index subarr range

    missing_vals = [city for city in parent1 if tuple(city) not in visited]
    missing_iter = iter(missing_vals)
    #print("missing_vals: ", missing_vals)
    #print("visited, before loop: ", visited)
    for i in range(n - 1): # keeps the last index of child blank
        #print("child: ", child)
        if child[i] is None:
            city_tuple = tuple(parent2[i])
            #print("city_tuple: ", city_tuple)
            if city_tuple not in visited:
                #print("city_tuple not in visited")
                child[i] = parent2[i]
                visited[city_tuple] = True
                
            else:
                while True:
                    next_city = next(missing_iter)
                    #print("city_tuple in visited, next_city is: ", next_city)

                    if tuple(next_city) not in visited:
                        child[i] = next_city
                        visited[tuple(next_city)] = True
                        break
        #print("visited, end of loop iter: ", visited)

    child[-1] = child[0]

    return child


def make_super_child(parents_list, start_index, end_index):
    if len(parents_list) == 1:
        return crossover(parents_list[0][0], 
                         parents_list[0][1], 
                         start_index, end_index)
    children = []
    for parent1, parent2 in parents_list:
        child = crossover(parent1, parent2, start_index, end_index)
        children.append(child)
    if len(children) > 1 and len(children) % 2 == 1: 
        children.pop()
    return make_super_child(get_rand_pairs(children), start_index, end_index)

def make_output(path):
    dist = str(calc_path_distance(path))
    string_list = list(map(lambda lst: " ".join(map(str, lst)), path))
    with open("output.txt", "w") as file:
        file.write(dist + "\n")  
        file.writelines((city + "\n") for city in string_list[:-1])
        file.write(string_list[-1])

with open("input.txt", "r") as input:
    num_cities = int(input.readline().strip())
    locations = []
    for line in input:
        city_location = [int(coord) for coord in line.split()]
        locations.append(city_location)


init_pop_size = 2000 if math.factorial(num_cities) >= 2000 else math.factorial(num_cities)
print("INITIAL POPULATION SIZE: ", init_pop_size)

# start 25% of the way through the array
start_index = num_cities // 4 if num_cities > 3 else 1
#end 75% of the way through the array
end_index = start_index * 2 if num_cities < 5 else start_index * 3

init_pop = create_init_population(init_pop_size, locations)
# print("INITIAL POPULATION:")
# [print(x) for x in init_pop]

rank_list = make_rank_list(init_pop)
# print("RANK LIST:")
# [print(x) for x in rank_list]

mating_pool = create_mating_pool(init_pop, rank_list)
# print("MATING POOL:")
# [print(x) for x in mating_pool]

super_child = make_super_child(get_rand_pairs(mating_pool), start_index, end_index)

print("CHILD:\n", super_child)
make_output(super_child)

