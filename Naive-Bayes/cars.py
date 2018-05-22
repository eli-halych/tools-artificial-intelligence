from functools import reduce


def load_dataset(trainFile, testFile, trainingSet=[], y=[], testSet=[]):
    # train.txt
    with open(trainFile, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        trainDataset = list(lines)
        for x in range(len(trainDataset)):
            trainingSet.append(trainDataset[x][:-1])
            y.append(trainDataset[x][-1])

    # test.txt
    with open(testFile, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        testDataset = list(lines)
        for x in range(len(testDataset)):
            # for y in range(6):
            #     testDataset[x][y] = float(testDataset[x][y])
            testSet.append(testDataset[x])


def count_outputs(y):
    unique_list = []
    for o in y:

        if not any(o in sublist for sublist in unique_list):
            count = 0
            for o2 in y:
                if o2 == o:
                    count += 1
            temp = [o, count]
            unique_list.append(temp)

    return unique_list


def class_prior_probability(y, length_all):
    return y / length_all


def likelihood(testCase, all_y, trainingSet, y, given_y, class_att):
    propabilities = []
    # att_count = []
    print(len(testCase))
    for i in range(len(testCase)):
        counter = 0
        for j in range(len(trainingSet)):
            if (trainingSet[j][i] == testCase[i]) and (given_y == y[j]):
                counter += 1
        # att_count.append([testCase[i], counter])
        propabilities.append((counter + 1) / (all_y + class_att[i]))
    # print(att_count, given_y)

    return reduce(lambda x, y: x * y, propabilities)


def evidence(testCase, trainingSet):
    counter = 0
    for x in trainingSet:
        # print(x, testCase)
        if x == testCase:
            print(x)
            counter += 1
    return counter / len(trainingSet)


def find_class_attributes(trainingSet):
    unique = [0 for i in range(len(trainingSet[0]))]
    all_values = [[] for i in range(len(trainingSet[0]))]

    for i in range(len(trainingSet[0])):
        for j in range(len(trainingSet)):
            if trainingSet[j][i] not in all_values[i]:
                all_values[i].append(trainingSet[j][i])

    for i in range(len(all_values)):
        unique[i] = len(all_values[i])

    return unique


def naive_bayes(trainingSet, y, testCase):
    class_att = find_class_attributes(trainingSet)

    num_of_y = count_outputs(y)  # [y, amount]
    P_y = []  # [y, probability]
    for yy in num_of_y:
        P_y.append([yy[0], class_prior_probability(yy[1], len(trainingSet))])
    print("  Class prior probabilities P(Y):\t " + str(P_y))

    likelihoods = []
    for i in range(len(num_of_y)):
        likelihoods.append(likelihood(testCase[:-1], num_of_y[i][1], trainingSet, y, num_of_y[i][0], class_att))
        # print(num_of_y[i])
    print("  Likelihoods P(X|Y):\t\t\t\t " + str(likelihoods))

    numerators = []
    for i in range(len(P_y)):
        numerators.append([likelihoods[i] * P_y[i][1], P_y[i][0]])
    print("  Posterior probabilities P(Y|X):\t " + str(numerators))

    return max(numerators)


def main():
    trainingSet = []
    y = []
    testSet = []
    load_dataset('car_bayes/training', 'car_bayes/test', trainingSet, y, testSet)

    correct_count = 0

    # PRELOADED CASES
    for i in range(len(testSet)):
        print(str(i) + "-----------------------------------------")
        result = naive_bayes(trainingSet, y, testSet[i])
        print("~~~~~~~~~~")
        correct = False

        if testSet[i][-1] == result[1]:
            correct = True
            correct_count += 1

        print(
            "Test case:\t\t\t\t\t\t\t " + str(testSet[i])
            + " \n  Result:\t\t\t\t\t\t\t "
            + "posterior probability: " + str(result[0]) + " \n\t\t\t\t\t\t\t\t\t "
            + "category: " + str(result[1])
            + "\n  Correct:\t\t\t\t\t\t\t " + str(correct)
        )
        print("-----------------------------------------\n\n\n")
    print("Accuracy for the whole test set = " + str(correct_count / len(testSet) * 100.0) + "%")

    # CUSTOM CASES
    while (True):
        custom_input = input("Do you want to test a custom input? [y/n]")
        if custom_input == "y" or custom_input == "Y":
            correct = False
            case = []
            print("EXAMPLE: [vhigh, vhigh, 2, more, small, high, unacc]")
            case.append(input("Enter the buying price [vhigh]: "))
            case.append(input("Enter the price of maintenance [vhigh]: "))
            case.append(input("Enter the number of doors [2]: "))
            case.append(input("Enter the passenger capacity [more]: "))
            case.append(input("Enter the the size of luggage boot [small]: "))
            case.append(input("Enter estimated safety [high]: "))
            case.append(input("Enter acceptability [unacc]: "))
            result = naive_bayes(trainingSet, y, case)
            if case[-1] == result[1]:
                correct = True

            print(
                "Test case:\t\t\t\t\t\t\t " + str(case)
                + " \n  Result:\t\t\t\t\t\t\t "
                + "posterior probability: " + str(result[0]) + " \n\t\t\t\t\t\t\t\t\t "
                + "category: " + str(result[1])
                + "\n  Correct:\t\t\t\t\t\t\t " + str(correct)
            )
            print("-----------------------------------------\n\n\n")
        if custom_input == "n" or custom_input == "N":
            break


main()
