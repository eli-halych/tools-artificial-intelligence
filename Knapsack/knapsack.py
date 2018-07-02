import math
import time



def bin_vector(num):
    return bin(num)

def bit_set(num, pos):
    return (num >> pos) & 1


def get_weight(vector, items, n):
    weight = 0
    for i in range(n):
        weight += items[i][1] * bit_set(vector, i)
    return weight


def get_feasible(n, items, capacity):
    feasible_vectors = []
    max = 2**n-1
    i = 0
    print(max)
    while i < max:
        if i % 1000000 == 0:
            print(i)
        if get_weight(i, items, n) <= capacity:
            feasible_vectors.append(i)
        i += 1
    return feasible_vectors


def get_value(vector, items, n):
    value = 0
    for i in range(n):
        value += items[i][2] * bit_set(vector, i)
    return value


def get_optimal(indicator_vectors, items, n):
    best_value = 0
    best_vector = 0

    for vector in indicator_vectors:
        value = get_value(vector, items, n)
        if value <= best_value:
            continue
        best_vector = vector
        best_value = value

    return bin_vector(best_vector), 0, best_value


def knapsack_brute_force(items, max_weight):
    n = len(items)
    capacity = max_weight

    indicator_vectors = get_feasible(n, items, capacity)
    print(indicator_vectors)
    knapsack, best_weight, best_value = get_optimal(indicator_vectors, items, n)

    return knapsack, best_weight, best_value


def build_items(n):
    from random import random
    res = []
    for i in range(n):
        res.append((i, 1 + int(9 * random()), 1 + int(9 * random())))
    return res

start_time = time.time()
n = 19  # number of items
items = build_items(n)
max_weight = 8
knapsack, opt_wt, opt_val = knapsack_brute_force(items, max_weight)

print("------------------------------------------------------------------------")
print("Items(id, weight, value): ", items)
print("\n\t\t\t  Max weight: ", max_weight)
print("\t\t\t\tKnapsack: ", knapsack)
print("\t\t  Optimal weight: ", opt_wt)
print("\t\t  Optimal value: ", opt_val)
print("\t\t  Execution time: %s seconds" % (time.time() - start_time))
print("------------------------------------------------------------------------")
