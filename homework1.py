import random
import math 
import bisect

def create_init_population(size, cities):
    """ 
    size = size of population to create
    cities = list of city locations parsed from input
    returns: list of paths (roundtrip -- ends with starting city) of length = size
    """
    paths = []
    orders_tried = {}
    city_indices = list(range(len(cities) - 1))
    
    for _ in range(size):
        city_order = city_indices.copy()

        while True: # ensures no duplicate paths
            random.seed()  # Reseed with a system-generated random value
            random.shuffle(city_order)
            if tuple(city_order) not in orders_tried:
                orders_tried[tuple(city_order)] = None    
                break
            
        city_order.append(city_order[0]) # add starting city to end of list
        path = [cities[j] for j in city_order] # populate w cities in city_order
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
    
def create_mating_pool(population, ranklist):
    pass

with open("input.txt", "r") as input:
    num_cities = int(input.readline().strip())
    locations = []
    for line in input:
        city_location = [int(coord) for coord in line.split()]
        locations.append(city_location)

init_pop = create_init_population(4, locations)
[print(x) for x in init_pop]
rank_list = make_rank_list(init_pop)
[print(x) for x in rank_list]