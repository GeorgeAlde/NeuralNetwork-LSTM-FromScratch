import numpy as np
from random import choice
from net import Network
from data import get_data, get_answers


def softmax(x1, x2): #returns softmax of x1
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
    surv_count_correct = 0
    surv_count = 0
    for i in range(len(mydata)):
        out = net.forward(mydata[i])
        correct_index = survival[i][1]
        softout = softmax(out[0], out[1]),softmax(out[1], out[0])

        L = loss(softout, correct_index)

        d_y = [0,0]
        for j in range(len(softout)):
            if survival[i][1] == j:
                d_y[j] = softout[j] - 1
            else:
                d_y[j] = softout[j]
        for k in range(2):
            net.backprop(d_y[k], k, learn_rate=0.0005, gradient_num=100)
        if np.argmax(softout) == correct_index:
            num_correct += 1
        if np.argmax(softout) == correct_index:
            if correct_index == 1:
                surv_count_correct += 1
            last100 += 1
        if np.argmax(softout) == 1:
            surv_count += 1
        count += 1


        if count == 100:
            print(f'{last100} out of last 100 correct')
            count = 0
            last100 = 0
    print(surv_count_correct, surv_count)
    count = 0
    for i in range(len(survival)):
        if (survival[i][1]) == 1:
            count += 1
    print(count, len(survival))
train()
