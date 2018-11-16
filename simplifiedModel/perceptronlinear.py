import numpy as np
from copy import copy, deepcopy


class PerceptronLinear:
    def __init__(self, weights_number, epsilon):
        # Biais weight
        self.weights_number = weights_number + 1
        self.weights = []
        self.generate_random_weight()
        self.epsilon = epsilon

    def generate_random_weight(self):
        self.weights = np.random.random(self.weights_number)

    def compute_y(self, inputs):
        result = 0.0
        for i in range(0, self.weights_number):
            result += inputs[i] * self.weights[i]
        return result

    def train_one_example(self, inputs, expected):
        inputs_treated = deepcopy(inputs)
        inputs_treated.insert(0, 1)
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] + self.epsilon * (expected - self.compute_y(inputs_treated)) * inputs_treated[i]

    def train(self, inputs, expected):
        inputs_treated = deepcopy(inputs)
        # Biais inputs
        for biais_index in range(0, len(inputs_treated)):
            inputs_treated[biais_index].insert(0, 1)

        for i in range(0, len(self.weights)):
            temp_sum = 0
            for j in range(0, len(expected)):
                temp_sum += (expected[j] - self.compute_y(inputs_treated[j])) * inputs_treated[j][i]
            self.weights[i] = self.weights[i] + self.epsilon * temp_sum

    def give_answer(self, inputs):
        inputs.insert(0, 1)
        return self.compute_y(inputs)
