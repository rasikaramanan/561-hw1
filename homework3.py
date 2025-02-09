import random
import math 
import bisect
import warnings 
import numpy as np
import time
import functools

# Dictionary to store function call counts and total execution time
profile_data = {}

def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        func_name = func.__name__

        # Update profiling data
        if func_name not in profile_data:
            profile_data[func_name] = {"calls": 0, "total_time": 0.0}
        
        profile_data[func_name]["calls"] += 1
        profile_data[func_name]["total_time"] += elapsed_time

        return result

    return wrapper


@profile
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
        while True: 
            random.seed()  # Reseed with a system-generated random value
            random.shuffle(city_order)
            if tuple(city_order) not in orders_tried:
                orders_tried[tuple(city_order)] = None    
                break

        city_order.append(city_order[0]) # add starting city to end of list
        path = [cities[j] for j in city_order] 
        paths.append(path)
    
    return paths


@profile
def calc_path_distance(path):
    path_arr = np.array(path)  
    
    path_arr = path_arr[:-1]
    
    shifted_path = np.roll(path_arr, shift=-1, axis=0)  # Shifted version to compute distances
    distances = np.linalg.norm(shifted_path - path_arr, axis=1)  # Vectorized Euclidean distance
    
    return np.sum(distances)

@profile
def make_rank_list(init_pop):
    rank_list = []
    for index, path in enumerate(init_pop):
        entry = (calc_path_distance(path), index)
        # will automatically sort by entry's val at index 0
        bisect.insort(rank_list, entry)
    return rank_list


@profile
def get_top_10_percent(num_pop):
    top10percent = num_pop // 10 
    if top10percent == 0: # 3 cities in init pop
        top10percent += 2
    elif top10percent % 2 == 1:
        top10percent += 1
    return top10percent

@profile
def get_non_elite_percent(num_pop):
    num_mp = num_pop // 10 * 4
    if num_mp == 0:
        num_mp = 2
    return num_mp

@profile
def create_mating_pool(population, rank_list):

    # ensure the top10percent is an even num
    top10percent = get_top_10_percent(len(rank_list)) 
    
    # 40% of population to be selected for mating from roulette wheel
    mp_rest = get_non_elite_percent(len(rank_list))

    # store the top 10 percent of population in top10_ranked, rest stays in rank_list
    top10_ranked, rank_list = rank_list[:top10percent], rank_list[top10percent:]

    population_dict = {index: route for index, route in enumerate(population)}

    # mating_pool is a list of paths, init with top10_ranked
    mating_pool = [population_dict[index] for (_, index) in top10_ranked]

    parent_indices = roulette_wheel_selection(rank_list, mp_rest)
    mating_pool += [population_dict[index] for index in parent_indices]
    return mating_pool

@profile
def roulette_wheel_selection(rank_list, num_to_select):
    # Invert fitness scores (lower is better, so higher probability)
    inv_fitness_rank_list = []
    sum_inv_fitness = 0.0

    for dist, index in rank_list:
        inv_fit = 1.0 / dist
        inv_fitness_rank_list.append((inv_fit, index))
        sum_inv_fitness += inv_fit
    
    parent_indices = []
    
    while len(parent_indices) < num_to_select: 
        rand_num = random.uniform(0, sum_inv_fitness)
        sum_pop = 0
        for score, index in inv_fitness_rank_list:
            sum_pop += score
            if rand_num < sum_pop:
                if index not in parent_indices: # ensures duplicate parents do not occur
                    parent_indices.append(index)
                break
    return parent_indices     

@profile
def tournament_selection(rank_list, num_to_select, tournament_size=3):
    parent_indices = []

    while len(parent_indices) < num_to_select:
        # Randomly select tournament_size candidates from rank_list
        tournament = random.sample(rank_list, tournament_size)
        
        # Choose the best (smallest dist) among them
        best_candidate = min(tournament, key=lambda x: x[0])  
        
        if best_candidate[1] not in parent_indices:  # Avoid duplicates
            parent_indices.append(best_candidate[1])

    return parent_indices

@profile
def get_rand_pairs(mating_pool):
    if len(mating_pool) % 2 == 1:
        mating_pool.pop()

    random.shuffle(mating_pool)
    return list(zip(mating_pool[::2], mating_pool[1::2]))

@profile
def crossover(parent1, parent2, start, end):

    size = len(parent1) - 1 # exclude last city 
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2)) 
    child[start:end] = parent1[start:end]

    remaining_cities = [city for city in parent2 if city not in child]
    insert_pos = 0
    for i in range(size): # keeps the last index of child blank
        if child[i] is None:
                child[i] = remaining_cities[insert_pos]
                insert_pos += 1

    child.append(child[0])
    return child

@profile
def mutate(path):
    num_cities = len(path) - 1 
    if num_cities < 4:  
        return path

    # Randomly select two indices (excluding first and last)
    i, j = sorted(random.sample(range(1, num_cities), 2))

    # Calc orig dist
    original_dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) + \
                    np.linalg.norm(np.array(path[j]) - np.array(path[j+1]))

    # Calc new dist
    new_dist = np.linalg.norm(np.array(path[j]) - np.array(path[i-1])) + \
               np.linalg.norm(np.array(path[i]) - np.array(path[j+1]))

    # Swap only if it reduces dist
    if new_dist < original_dist:
        path[i], path[j] = path[j], path[i]
        # print("IN MUTATE, path dist improved by: ", original_dist - new_dist)

    return path

@profile
def two_opt(path, num_improvements = 50):
    """Applies the 2-opt local optimization heuristic to improve a given path."""
    curr_path = path.copy()
    num_cities = len(path)

    imprvmnts_made = 0
    for segment_length in range(num_cities - 3, 1, -1):  # Start with largest swaps
        for i in range(1, num_cities - segment_length - 1):  # Skip first and last city
            j = i + segment_length 

            if j >= num_cities - 1:  # Prevent out-of-bounds errors
                break

            prev_dist = np.linalg.norm(np.array(curr_path[i]) - np.array(curr_path[i-1])) + \
                        np.linalg.norm(np.array(curr_path[j-1]) - np.array(curr_path[j]))
            new_dist = np.linalg.norm(np.array(curr_path[j-1]) - np.array(curr_path[i-1])) + \
                        np.linalg.norm(np.array(curr_path[i]) - np.array(curr_path[j]))
            
            if new_dist < prev_dist:
                curr_path[i:j] = curr_path[i:j][::-1]
                imprvmnts_made += 1
                break #for a given segment length, only do one successful swap
        if imprvmnts_made >= num_improvements:
            break
    return curr_path
            
@profile
def make_super_child(parents_list, start_index, end_index):
    if len(parents_list) == 1: # basecase 
        crossed = crossover(parents_list[0][0], 
                         parents_list[0][1], 
                         start_index, end_index)
        return two_opt(crossed)
    
    children = []
    for parent1, parent2 in parents_list:
        crossed = crossover(parent1, parent2, start_index, end_index)
        children.append(two_opt(crossed))
    if len(children) > 1 and len(children) % 2 == 1: 
        children.pop()
    return make_super_child(get_rand_pairs(children), start_index, end_index)

def make_output(path):
    dist = str(calc_path_distance(path))
    print("PATH DIST FOR OUTPUT: ", dist)
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


init_pop_size = 4000 if math.factorial(num_cities) >= 4000 else math.factorial(num_cities)
print("INITIAL POPULATION SIZE: ", init_pop_size)

# start 25% of the way through the array
start_index = num_cities // 4 if num_cities > 3 else 1

#end 75% of the way through the array
end_index = start_index * 2 if num_cities < 5 else start_index * 3

init_pop = create_init_population(init_pop_size, locations)

rank_list = make_rank_list(init_pop)
print("SHORTEST PATH DIST FROM INIT POPULATION: ", rank_list[0][0])
best_init_dist = rank_list[0][0]
best_init_path = init_pop[rank_list[0][1]]

mating_pool = create_mating_pool(init_pop, rank_list)

super_child = make_super_child(get_rand_pairs(mating_pool), start_index, end_index)
dist_sc = calc_path_distance(super_child)

print("SUPER CHILD PATH DIST: ", dist_sc)

super_child = super_child if dist_sc < best_init_dist else best_init_path
    
make_output(super_child)

def print_profiling_results():
    total_runtime = sum(data["total_time"] for data in profile_data.values())
    print(f"\n=== Total Runtime: {total_runtime:.4f} seconds ===")
    for func, data in sorted(profile_data.items(), key=lambda x: x[1]["total_time"], reverse=True):
        calls = data["calls"]
        total_time = data["total_time"]
        percentage = (total_time / total_runtime) * 100 if total_runtime else 0
        print(f"{func}: {calls} calls, {total_time:.4f} sec total, {percentage:.2f}% of runtime")

print_profiling_results()