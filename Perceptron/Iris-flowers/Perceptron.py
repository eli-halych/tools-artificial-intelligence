# perceptron.py
import numpy as np


class PerceptronTask(object):
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate
        self.niter = niter

    def loadDataset(self, train_file, test_file, training_set=[], test_set=[]):
        # train.txt
        with open(train_file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            train_dataset = list(lines)
            for x in range(len(train_dataset) - 1):
                for y in range(4):
                    train_dataset[x][y] = float(train_dataset[x][y])
                training_set.append(train_dataset[x])

        # test.txt
        with open(test_file, 'r') as in_file:
            stripped = (line.strip() for line in in_file)
            lines = (line.split(",") for line in stripped if line)
            test_dataset = list(lines)
            for x in range(len(test_dataset) - 1):
                for y in range(4):
                    test_dataset[x][y] = float(test_dataset[x][y])
                test_set.append(test_dataset[x])

    def output(self, training_set, weights, threshold):
        sum = 0

        for j in range(4):
            sum += weights[j] * training_set[j]

        return sum >= threshold

    def update_weights(self, desired_y, y, X, alpha, weights, threshold):

        w = list(weights)
        w.append(threshold)

        x = list(X)
        x.append(-1)

        mult = (float(desired_y) - float(y)) * alpha
        for i in range(4):
            w[i] += mult * x[i]
        print(w)

        return w


def splitTrainandTest(training_set, test_set, X_train, y_train, X_test, y_test):
    for i in range(len(training_set)):
        X_train.append(training_set[i][0:4])
        y_train.append(np.where(training_set[i][4] == 'Iris-virginica', 0, 1))

    for i in range(len(test_set)):
        X_test.append(test_set[i][0:4])
        y_test.append(np.where(test_set[i][4] == 'Iris-virginica', 0, 1))

    pass


def main():
    perceptron = PerceptronTask()

    # prepare data
    training_set = []
    test_set = []
    perceptron.loadDataset('training.txt', 'test.txt', training_set, test_set)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    splitTrainandTest(training_set, test_set, X_train, y_train, X_test, y_test)

    weights = [0.3, 0.2, 0.4, 0.2]
    threshold = 0.7
    alpha = float(input("Alpha: "))

    for i in range(perceptron.niter):
        for j in range(len(X_train)):
            output = int(perceptron.output(X_train[j], weights, threshold))
            if output == y_train[j]:
                continue
            else:
                new_weights_thetha = perceptron.update_weights(y_train[j], output, X_train[j], alpha, weights, threshold)
                for k in range(5):
                    if k < 4:
                        weights[k] = new_weights_thetha[k]
                    if k == 4:
                        threshold = new_weights_thetha[k]

    correct = 0
    all = 0
    for i in range(len(X_test)):
        output = int(perceptron.output(X_test[i], weights, threshold))
        if output == y_test[i]:
            correct += 1
        all += 1

        print("Expected=" + str(y_test[i]) + "     Predicted=" + str(output))

    print("Accuracy=" + str(float(correct / all) * 100) + "%")


main()
