import numpy as np
from net import Network
from data import get_data, get_answers


def softmax(x1, x2): #returns softmax of x1
    return np.exp(x1) / (np.exp(x1) + np.exp(x2))

def loss(output, correct_index):
    return -np.log(output[correct_index])


mydata = get_data()
survival = get_answers()
net = Network()
for i in range(len(mydata)):
    out = net.forward(mydata[i])
    softout = softmax(out[0], out[1]),softmax(out[1], out[0])
    L = loss(softout, survival[i][1])
    print(L)