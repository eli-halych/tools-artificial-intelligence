import csv
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
def likelihood(testCase, all_y, trainingSet, y, given_y):
    propabilities = []
    for i in range(len(testCase)):
        counter = 0
        for j in range(len(trainingSet)):
            if (trainingSet[j][i] == testCase[i]) and (given_y == y[j]):
                counter+=1
        propabilities.append(counter/all_y)

    return reduce(lambda x, y: x*y, propabilities)
def evidence(testCase, trainingSet):
    counter = 0
    for x in trainingSet:
        # print(x, testCase)
        if x == testCase:
            print(x)
            counter+=1
    return counter/len(trainingSet)


def naive_bayes(trainingSet, y, testCase):

    num_of_y = count_outputs(y) # [y, amount]
    P_y = [] # [y, probability]

    for yy in num_of_y:
        P_y.append([yy[0], class_prior_probability(yy[1], len(trainingSet))])

    likelihoods = []
    for i in range(len(num_of_y)):
        likelihoods.append(likelihood(testCase[:-1], num_of_y[i][1], trainingSet, y, num_of_y[i][0]))

    numerators = []
    for i in range(len(P_y)):
        numerators.append([likelihoods[i]*P_y[i][1], P_y[i][0]])

    return max(numerators)





def main():
    trainingSet = []
    y = []
    testSet = []
    load_dataset('car_bayes/training', 'car_bayes/test', trainingSet, y, testSet)

    correct_count = 0

    for i in range(len(testSet)):

        result = naive_bayes(trainingSet, y, testSet[i])
        correct = False

        if testSet[i][-1] == result[1]:
            correct = True
            correct_count+=1

        print("Test case: " + str(testSet[i]) + " \n Result: " + "posterior probability: " + str(result[0]) + " ||| "
              + "category: " + str(result[1]) + "\n Correct: " + str(correct))
        print("-----------------------------------------")
    print("Accuracy for the whole test set = "+str(correct_count/len(testSet)*100.0)+"%")



main()
