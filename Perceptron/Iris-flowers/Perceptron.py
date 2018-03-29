# perceptron.py
import numpy as np


class Perceptron(object):
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        """Fit training data
      X : Training vectors, X.shape : [#samples, #features]
      y : Target values, y.shape : [#samples]
      """

        # weights
        self.weight = np.zeros(1 + X.shape[1])

        # Number of misclassifications
        self.errors = []  # Number of misclassifications

        for i in range(self.niter):
            err = 0
            for xi, target in zip(X, y):
                delta_w = self.rate * (target - self.predict(xi))
                self.weight[1:] += delta_w * xi
                self.weight[0] += delta_w
                err += int(delta_w != 0.0)
            self.errors.append(err)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


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
        sum -= threshold

        if sum < threshold:
            return 0
        else:
            return 1

    def update_weights(self, desired_output, output, training_set, alpha, weights, threshold):
        new_x = []
        new_w = []
        for x in range(3):
            new_x.append(0.0)
            new_w.append(0.0)

        temp = alpha*desired_output*output
        # print(tr)

        for j in range(3):
            # print(new_x[j])
            # print(new_w[j])
            new_x[j] = training_set[j] * temp
            new_w[j] = weights[j] - new_x[j]
        threshold = new_w[len(new_w)-1]
        return new_w



def main():
    perceptron = PerceptronTask()

    # prepare data
    training_set = []
    test_set = []
    perceptron.loadDataset('training.txt', 'test.txt', training_set, test_set)
    print('Train set: ' + repr(len(training_set)))
    print("Test set: " + repr(len(test_set)))

    weights = [0.3, 0.2, 0.4, 0.2]
    threshold = 0.7

    for i in range(perceptron.niter):
        output = perceptron.output(training_set[i], weights, threshold)
        weights = perceptron.update_weights(1, output, training_set[i], 0.5, weights, threshold) # refactor the weights and threshold args
        print(weights)

    # print(output)


main()
