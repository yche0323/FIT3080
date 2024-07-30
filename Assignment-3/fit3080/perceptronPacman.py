# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import matplotlib.pyplot as plt
# from data_classifier_new import convertToArray
import math
import numpy as np

PRINT = True


class SingleLayerPerceptronPacman():

    def __init__(self, num_weights=5, num_iterations=20, learning_rate=1):

        # weight initialization
        # model parameters initialization

        # initialise all features to be 1s
        self.weights = np.ones(num_weights)

        self.max_iterations = num_iterations
        self.learning_rate = learning_rate


    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and compute
        the dot product of the weights of your perceptron with the values of features.

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        Then the result of this computation should be passed through your activation function

        For example if x=feature_vector, and ReLU is the activation function
        this function should compute ReLU(x dot weights)
        """

        "*** YOUR CODE HERE ***"
        perceptron_weights_sum = np.dot(feature_vector, self.weights)
        predicted_value = self.activation(perceptron_weights_sum)
        return predicted_value

    def activation(self, x):
        """
        Implement your chosen activation function here.
        """

        "*** YOUR CODE HERE ***"
        return 1/(1 + np.exp(-x))

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable.

        The data should be a 2D numpy array where the each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        "*** YOUR CODE HERE ***"
        true_predictions = 0

        for i in range(len(data)):
            curr_feature_vector = data[i]
            predicted_value = self.predict(curr_feature_vector)

            if predicted_value >= 0.5:
                classify_prediction = 1
            else:
                classify_prediction = 0

            if classify_prediction == labels[i]:
                true_predictions += 1

        prediction_accuracy = true_predictions / len(data)
        return prediction_accuracy


    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        "*** YOUR CODE HERE ***"
        for _ in range(self.max_iterations):
            predicted_values = []
            total_loss = 0.0
            for i in range(len(trainingData)):
                feature_vector = trainingData[i]
                true_value = trainingLabels[i]

                predicted_value = self.predict(feature_vector)
                loss = -1 * (true_value * np.log(predicted_value) + (1-predicted_value) * np.log(1-predicted_value))

                predicted_values.append(predicted_value)
                total_loss += loss

            total_loss /= len(trainingData)

            for w in range(len(self.weights)):
                weight_change = 0.0
                for i in range(len(trainingLabels)):
                    weight_change += (predicted_values[i] - trainingLabels[i]) * trainingData[i][w]

                self.weights[w] -= self.learning_rate * weight_change * float(1 / len(trainingLabels))


