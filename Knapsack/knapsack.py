import math


# here i run out of memory if #of items is 30
def powerset(items):
    res = [[]]
    for item in items:
        newset = [r + [item] for r in res]
        res.extend(newset)
    return res


def indicator_vector(num):
    return bin(num)[2:]


def characteristic_vector(items):
    res = []


def knapsack_brute_force(items, max_weight):
    knapsack = []
    n = len(items)
    best_weight = 0
    best_value = 0
    bin_range = range(0, int(math.pow(2, n)))
    bag = []

    for i in bin_range:
        b = indicator_vector(i)
        w_sum = 0
        v_sum = 0

        # ------------REVERSING INDICES FOR A BINARY NUMBER, testing
        # x = n - len(b)
        # # print("x", x)
        # for k in range(len(b)):
        #     current_index = ((len(b)+x)-k)-1
        #     if current_index > 0 and b != '0':
        #         print("index = ", current_index, "for binary = ", b)
        # print("------------------------")
        # ------------

        num_zeros_before = n - len(b)
        for j in range(len(b)):
            current_index = ((len(b) + num_zeros_before) - j) - 1
            # 0 is a useless case
            if b != '0':

                if b[j] == '1':
                    w_sum += items[current_index][1]
                    v_sum += items[current_index][2]

        if w_sum <= max_weight and v_sum > best_value and b != '0':
            best_weight = w_sum
            best_value = v_sum
            bag = []
            for e in range(len(b)):
                curr_ind = ((len(b) + num_zeros_before) - j) - 1
                if b[e] == '1':
                    bag.append(items[curr_ind])
            print("Probable knapsack:", bag)

        knapsack = bag

    return knapsack, best_weight, best_value


def build_items(n):
    from random import random
    res = []
    for i in range(n):
        res.append((i, 1 + int(9 * random()), 1 + int(9 * random())))
    return res


n = 10 # number of items
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
