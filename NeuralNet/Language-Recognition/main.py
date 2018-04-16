# THIS IMPLEMENTATION TRADES EFFICIENCY FOR UNDERSTANDABILITY

import numpy as np
import collections
import string
import random
import math

from matplotlib import pyplot as plt
#
# # T
# class Neuron(object):
#     def __init__(self, n_inputs):
#         self.n_inputs = n_inputs
#         self.set_weights( [ random.uniform(0, 1) for x in range(0, n_inputs) ] )
#         # print(self.weights)
#         self.E = 0.0
#
#     # def output(self, training_set, weights, threshold):
#     #     sum = 0
#     #
#     #     for j in range(4):
#     #         sum += weights[j] * training_set[j]
#     #
#     #     return sum >= threshold
#     #
#     # def update_weights(self, desired_y, y, X, alpha, weights, threshold):
#     #
#     #     w = list(weights)
#     #     w.append(threshold)
#     #
#     #     x = list(X)
#     #     x.append(-1)
#     #
#     #     mult = (float(desired_y) - float(y)) * alpha
#     #     for i in range(4):
#     #         w[i] += mult * x[i]
#     #     print(w)
#     #
#     #     return w
#
#     def sum(self, threshold):
#         self.net = np.dot(self.inputs[0:26], self.weights) - threshold
#         # print(self.net)
#
#     def act_unip(self):
#         try:
#             self.y = 1 / (1 + math.exp(-self.net*0.01))
#             # print(self.y)
#         except OverflowError:
#             return float("inf")
#     def act_bip(self):
#         try:
#             self.y = (2 / (1 + math.exp(-self.net*0.01))) - 1
#         except OverflowError:
#             return float("inf")
#
#     def err(self, E_max):
#         self.E += (1.0 - self.y)*(1.0 - self.y) / 2
#         print(self.E)
#
#         # return self.E
#
#     def der(self, output):
#         pass
#
#     def upd(self, rate):
#         x_new = []
#
#
#     def set_weights(self, weights):
#         self.weights = weights
#
#     def __str__(self):
#         return 'Weights: %s ' % (str(self.weights[:]))
#
#
#     def loadInputs(self, x):
#         self.inputs = x
#         # print(self.inputs)
#     def loadOutputs(self, y_train):
#         self.desired = y_train
#     def setLang(self, lang):
#         self.lang = lang
#
#     def toFloat(self, x):
#         return float("{0:.2f}".format(x))
#
#
# # done
# class NeuronLayer(object):
#     def __init__(self, n_neurons, n_inputs):
#         self.n_neurons = n_neurons
#         self.neurons = [Neuron(n_inputs) for _ in range(0, self.n_neurons)]
#         # print(self.neurons[0].weights)
#
#     def __str__(self):
#         return 'Layer:\n\t' + '\n\t'.join([str(neuron) for neuron in self.neurons]) + ''
#
# # TODO
# class NeuralNetwork(object):
#     def __init__(self, n_inputs, n_outputs, n_neurons_to_hl, n_hidden_layers):
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.n_hidden_layers = n_hidden_layers
#         self.n_neurons_to_hl = n_neurons_to_hl
#
#         # Do not touch
#         self._create_network()
#         self._n_weights = None
#         # end
#     def _create_network(self):
#         if self.n_hidden_layers > 0:
#             # create the first layer
#             self.layers = [NeuronLayer(self.n_neurons_to_hl, self.n_inputs)]
#
#             # create hidden layers
#             self.layers += [NeuronLayer(self.n_neurons_to_hl, self.n_neurons_to_hl) for _ in
#                             range(0, self.n_hidden_layers)]
#
#             # hidden-to-output layer
#             self.layers += [NeuronLayer(self.n_outputs, self.n_neurons_to_hl)]
#         else:
#             # If we don't require hidden layers
#             self.layers = [NeuronLayer(self.n_outputs, self.n_inputs)]
#     def upd(self, inputs):
#         for layer in self.layers:
#             outputs = []
#             for neuron in layer.neurons:
#                 tot = neuron.sum(inputs)
#                 outputs.append(self.act_unip(tot))
#             # inputs = outputs
#         return outputs
#
#     # def update(self, inputs):
#     #     assert len(inputs) == self.n_inputs, "Incorrect amount of inputs."
#     #
#     #     for layer in self.layers:
#     #         outputs = []
#     #         for neuron in layer.neurons:
#     #             tot =
#     #             outputs.append(self.sigmoid(tot))
#     #         inputs = outputs
#     #     return outputs
#
#     def __str__(self):
#         return '\n'.join([str(i + 1) + ' ' + str(layer) for i, layer in enumerate(self.layers)])



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
        self.E = 0
        # print(self.weights)
        # print(type(self.weights))
        # print(type(self.thresholds))

    def sigmoid(self, net):
        return 1 / (1 + math.exp(-net * 0.01))
    def derivative(self, output):
        return output*(1 - output)

    def update_weights(self, desired_output, output, input_vector):
        self.E += 0.5 * (desired_output - output) * (desired_output - output)
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


    def feedworward_train(self, input_vector, desired_output):
        net = np.dot(self.weights, input_vector)
        output = self.sigmoid(net)
        if desired_output == self.lang and output != 1.0:
            self.update_weights(1.0, output, input_vector)
        if desired_output != self.lang and output != 0.0:
            self.update_weights(0.0, output, input_vector)
        # print(net, output)

    def feedforward_test(self, input_vector):
        net = np.dot(self.weights, input_vector)
        return net



    # def feedforward(self, input_vector, output_vector):
    #     a = []
    #     for b, w in zip(self.thresholds, self.weights):
    #         net = np.dot(w, input_vector.T) - b
    #         output = self.sigmoid(net)
    #         print(output)
    #         a.append(output)
    #
    #     a = np.matrix(a).T
    #     # print(a)


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


    en = Neuron("en", 100)
    pl = Neuron("pl", 100)
    de = Neuron("de", 100)

    for j in range(100):
        for i in range(len(X_train)):
            en.feedworward_train(X_train[i], y_train[i])
            pl.feedworward_train(X_train[i], y_train[i])
            de.feedworward_train(X_train[i], y_train[i])
        #     print("----------" + str(i))
        # print("///////////////////////")

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




    # en_neuron = neural_network.layers[0].neurons[0]
    # pl_neuron = neural_network.layers[0].neurons[1]
    # de_neuron = neural_network.layers[0].neurons[2]
    #
    # en_neuron.setLang("en")
    # pl_neuron.setLang("pl")
    # de_neuron.setLang("de")
    #
    # en_neuron.loadOutputs(y_train)
    # pl_neuron.loadOutputs(y_train)
    # de_neuron.loadOutputs(y_train)

    # print(en_neuron.weights)

    # for x in X_train:
    #     print("en")
    #     en_neuron.loadInputs(x)
    #     en_neuron.sum(threshold)
    #     en_neuron.act_unip()
    #     en_neuron.err(E_max)
    #
    #     print("---------------")






main()
