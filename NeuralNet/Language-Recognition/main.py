# THIS IMPLEMENTATION TRADES EFFICIENCY FOR UNDERSTANDABILITY

import collections
import math
import string

import numpy as np



def loadDataset(train_dir, test_dir, languages, num_of_files, training_set=[], test_set=[], case_sensitive=False):
    # training
    for i in range(len(languages)):
        for j in range(num_of_files):
            with open(train_dir+languages[i] + "/" + str(j + 1) + ".txt") as in_file:
                original_text = in_file.read()
                if case_sensitive:
                    alphabet = string.ascii_letters
                    text = original_text
                else:
                    alphabet = string.ascii_lowercase
                    text = original_text.lower()
                alphabet_list = list(alphabet)

                counts = collections.Counter(c for c in text if c in alphabet_list) # ignoring special letters like ą, ę, etc.

                letters_count = []
                for letter in alphabet:
                    letters_count.append(int(counts[letter]/26))

                letters_count.append(languages[i])
                training_set.append(letters_count)


    # test
    for i in range(len(languages)):
        for j in range(3):
            with open(test_dir+languages[i] + "/" + str(j+1) + ".txt") as in_file:
                original_text = in_file.read()
                if case_sensitive:
                    alphabet = string.ascii_letters
                    text = original_text
                else:
                    alphabet = string.ascii_lowercase
                    text = original_text.lower()
                alphabet_list = list(alphabet)

                counts = collections.Counter(
                    c for c in text if c in alphabet_list)  # ignoring special letters like ą, ę, etc.

                letters_count = []
                for letter in alphabet:
                    letters_count.append(int(counts[letter] / 26))

                letters_count.append(languages[i])
                test_set.append(letters_count)
def splitTrainandTest(training_set, test_set, X_train, y_train, X_test, y_test):
    # training
    for i in range(len(training_set)):
        X_train.append(training_set[i][0:26])
        y_train.append(training_set[i][26])

    #test
    for i in range(len(test_set)):
        X_test.append(test_set[i][0:26])
        # y_test.append(test_set[i][26])





class Neuron():
    def __init__(self, lang, threshold):
        # self.thresholds = np.matrix('10, 20, 15').T
        self.weights = np.random.rand(1, 26)
        self.lang = lang
        self.threshold = threshold
        self.difference = 0
        # print(self.weights)
        # print(type(self.weights))
        # print(type(self.thresholds))

    def sigmoid(self, net):
        return 1 / (1 + math.exp(-net * 0.01))
    def derivative(self, output):
        return output*(1 - output)

    def update_weights(self, desired_output, output, input_vector):
        # self.E += 0.5 * (desired_output - output) * (desired_output - output)
        # print(self.E) # increases over time

        coefficient =  0.01*(desired_output-output)*self.derivative(output)
        new_X = [x * coefficient for x in input_vector]
        self.weights += new_X
        self.normalizae_weights()
        # print(self.weights)

    def normalizae_weights(self):
        length = 0.0
        for w in self.weights[0]:
            length += w*w
            # print(w)
        length = math.sqrt(length)

        for w in self.weights[0]:
            w /= length

        # print(self.weights)

    # def error(self, output, desired_output):
    #     self.error += 0.5*(desired_output - output)*(desired_output - output)

    def feedworward_train(self, input_vector, desired_output):
        self.net = np.dot(self.weights, input_vector)
        net = self.net #unnecessary

        self.output = self.sigmoid(net)
        output = self.output #unnecessary

        # self.E = 0/5*(math.pow(1.0))

        if desired_output == self.lang and output != 1.0:
            self.update_weights(1.0, output, input_vector)
            self.difference = (1.0 - output)
        if desired_output != self.lang and output != 0.0:
            self.update_weights(0.0, output, input_vector)
            self.difference = (0.0 - output)
        # print(net, output)

    def feedforward_test(self, input_vector):
        net = np.dot(self.weights, input_vector)
        return net



def main():

    train_path = "training/"
    test_path = "test/"
    lang = ["en", "pl", "de"]

    training_set = []
    test_set = []

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    loadDataset(train_path, test_path, lang, 10, training_set, test_set)
    splitTrainandTest(training_set, test_set, X_train, y_train, X_test, y_test)

    # print(X_test)

    # X_train = np.matrix(X_train)
    # y_train = np.matrix(y_train)
    # X_test = np.matrix(X_test)
    # y_test = np.matrix(y_test)


    Emax = 0.5
    E = 0

    en = Neuron("en", 100)
    pl = Neuron("pl", 100)
    de = Neuron("de", 100)

    # draft for Error implementation
    # outputs = [[None for _ in range(2)] for _ in range(3)]
    # outputs[0][0] = 1.0
    # outputs[0][1] = 0.2
    # print(outputs)

    while True:
        for i in range(len(X_train)):
            en.feedworward_train(X_train[i], y_train[i])
            pl.feedworward_train(X_train[i], y_train[i])
            de.feedworward_train(X_train[i], y_train[i])

            E += 0.5*((en.difference*en.difference)+(pl.difference*pl.difference)+(de.difference*de.difference))

            # print("----------" + str(i))
        print(E)
        if(E < Emax):
            break
        else: E = 0
        print("///////////////////////")


        # for the future Error implementation
        # if E < Emax:
        #     break
        # else:
        #     E = 0

    for i in range(len(X_test)):
        en_o = en.feedforward_test(X_test[i])
        pl_o = pl.feedforward_test(X_test[i])
        de_o = de.feedforward_test(X_test[i])
        if en_o > pl_o and en_o > de_o:
            print("english")
        if pl_o > en_o and pl_o > de_o:
            print("polish")
        if de_o > en_o and de_o > pl_o:
            print("german")






main()
