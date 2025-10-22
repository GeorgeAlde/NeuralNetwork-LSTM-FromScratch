import numpy as np

from data import get_data, get_answers
from net import Network


def softmax(x1, x2):  # returns softmax of x1
    return np.exp(x1) / (np.exp(x1) + np.exp(x2))


def loss(output, correct_index):
    if output[correct_index] != 0:
        return -np.log(output[correct_index])
    else:
        return 10


net = Network()


def train():
    num_correct = 0
    last100 = 0
    count = 0
    mydata = get_data()
    survival = get_answers()
    for i in range(len(mydata)):
        out = net.forward(mydata[i])
        correct_index = survival[i][1]
        softout = softmax(out[0], out[1]), softmax(out[1], out[0])

        L = loss(softout, correct_index)

        d_y = [0, 0]
        for j in range(len(softout)):
            if survival[i][1] == j:
                d_y[j] = softout[j] - 1
            else:
                d_y[j] = softout[j]
        for k in range(2):
            net.backprop(d_y[k], k, learn_rate=0.001)
        if np.argmax(softout) == correct_index:
            num_correct += 1
        if np.argmax(softout) == correct_index:
            last100 += 1
        count += 1

        if count == 100:
            print(f'{last100} out of last 100 correct')
            count = 0
            last100 = 0



train()
