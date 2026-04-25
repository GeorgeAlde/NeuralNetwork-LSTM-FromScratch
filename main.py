import numpy as np

from data import get_data, get_answers
from net import Network


def softmax(x1, x2):
    ex = np.exp([x1, x2])
    return ex / np.sum(ex)


def loss(output, correct_index):
    return -np.log(output[correct_index] + 1e-9)


net = Network()


def train():
    num_correct = 0
    last100 = 0
    count = 0

    mydata = get_data()
    survival = get_answers()

    for rep in range(10):

        for i in range(len(mydata)):

            out = net.forward(mydata[i])

            correct_index = int(survival[i][1])

            softout = softmax(out[0], out[1])

            d_y = softout.copy()
            d_y[correct_index] -= 1

            L = loss(softout, correct_index)

            net.backprop(d_y)

            if np.argmax(softout) == correct_index:
                num_correct += 1
                last100 += 1

            count += 1

            if count == 100:
                print(f'{last100} out of last 100 correct')
                count = 0
                last100 = 0


train()