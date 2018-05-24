import matplotlib
import matplotlib.pyplot as plt


def powerset(items):
    res = [[]]
    for item in items:
        newset = [r + [item] for r in res]
        res.extend(newset)
    return res

def indicator_vector(items):
    pass

def characteristic_vector(items):
    res = []

def knapsack_brute_force(items, max_weight):
    knapsack = []
    best_weight = 0
    best_value = 0
    for item in items:
        for j in range(len(item)):
            


        # set_weight = sum(map(weight, item_set))
        # set_value = sum(map(value, item_set))
        # if set_value > best_value and set_weight <= max_weight:
        #     best_weight = set_weight
        #     best_value = set_value
        #     knapsack = item_set
    return knapsack, best_weight, best_value


def weight(item):
    return item[1]


def value(item):
    return item[2]


def build_items(n):
    from random import random
    res = []
    for i in range(n):
        res.append((i, 1 + int(9 * random()), 1 + int(9 * random())))
    return res


# items = [(0,2,4), (1,5,3), (2,7,4), (3,3,5)]
n = 10
print("Computations =", 2**n)
items = build_items(n)
max_weight = 20
knapsack, opt_wt, opt_val = knapsack_brute_force(items, max_weight)

print("------------------------------------------------------------------------")
print("Items(id, weight, value): ", items)
print("\n\t\t\t  Max weight: ", max_weight)
print("\t\t\t\tKnapsack: ", knapsack)
print("\t\t  Optimal weight: ", opt_wt)
print("\t\t   Optimal value: ", opt_val)
print("------------------------------------------------------------------------")
