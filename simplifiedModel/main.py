import numpy as np
import matplotlib.pyplot as plt

from simplifiedModel.perceptronlinear import PerceptronLinear

if __name__ == "__main__":
    perceptron = PerceptronLinear(3, 0.1)
    print("Old weight : ", perceptron.weights)
    inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    expected = [0, 0, 0, 0, 0, 0, 1, 1]
    weight1 = []
    weight2 = []
    weight3 = []
    weight4 = []
    for i in range(0, 10000):
        # result.append(perceptron.give_answer(inputs))
        perceptron.train(inputs, expected)
        weight1.append(perceptron.weights[0])
        weight2.append(perceptron.weights[1])
        weight3.append(perceptron.weights[2])
        weight4.append(perceptron.weights[3])

    print("New weight : ", perceptron.weights)
    print("Truth Table : ")
    for i in range(0, len(inputs)):
        print(inputs[i], " | ", perceptron.give_answer(inputs[i]))
    plt.plot(weight1)
    plt.plot(weight2)
    plt.plot(weight3)
    plt.plot(weight4)

    plt.title("Output evolution")
    plt.show()
