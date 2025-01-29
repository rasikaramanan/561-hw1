import random
import math 
import bisect
import warnings 

def create_init_population(size, cities):
    """ 
    size = size of population to create
    cities = list of city locations parsed from input
    returns: list of paths (roundtrip -- ends with starting city) of length = size
    """

    if size > math.factorial(len(cities)):
        warnings.warn(f"Given size of {size} is greater than the number of permutations possible for {len(cities)} cities. Setting size to the max allowable value, {math.factorial(len(cities))}.")
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
    total_dist = 0.0

    # loop calcs dist between two cities
    for i in range(len(path) - 1): 
        city1 = path[i]
        city2 = path[i + 1]
        dist = math.sqrt(
            (city2[0] - city1[0])**2 + 
            (city2[1] - city1[1])**2 + 
            (city2[2] - city1[2])**2 
        )
        total_dist += dist
    return total_dist

def make_rank_list(init_pop):
    rank_list = []
    for index, path in enumerate(init_pop):
        entry = (calc_path_distance(path), index)
        # will automatically sort by entry's val at index 0
        bisect.insort(rank_list, entry)
    return rank_list
    
def create_mating_pool(population, rank_list):
    # Invert fitness scores (lower is better, so higher probability)
    inv_fitness_rank_list = []
    sum_inv_fitness = 0.0

    for dist, index in rank_list:
        inv_fit = 1.0 / dist
        inv_fitness_rank_list.append((inv_fit, index))
        sum_inv_fitness += inv_fit
    
    parent_indices = []
    
    while len(parent_indices) < 2: #SELECT MATING POOL OF SIZE = 2
        rand_num = random.uniform(0, sum_inv_fitness)
        #print("random number is: ", rand_num)
        sum_pop = 0
        for score, index in inv_fitness_rank_list:
            sum_pop += score
            if rand_num < sum_pop:
                if index not in parent_indices: # ensures duplicate parents do not occur
                    parent_indices.append(index)
                    print("PARENT ADDED: ", score, index)
                break 
    print("parent indices selected: ", parent_indices)
    population_dict = {index: route for index, route in enumerate(population)}
    mating_pool = [population_dict[index] for index in parent_indices]
    return mating_pool

with open("input.txt", "r") as input:
    num_cities = int(input.readline().strip())
    locations = []
    for line in input:
        city_location = [int(coord) for coord in line.split()]
        locations.append(city_location)

init_pop = create_init_population(50, locations)
print("INITIAL POPULATION:")
[print(x) for x in init_pop]

rank_list = make_rank_list(init_pop)
print("RANK LIST:")
[print(x) for x in rank_list]

mating_pool = create_mating_pool(init_pop, rank_list)
print("MATING POOL:")
[print(x) for x in mating_pool]

