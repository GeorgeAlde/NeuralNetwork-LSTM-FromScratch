import numpy as np
from numpy.random import randn


class Network:

    def __init__(self):
        self.Wxh = randn(7, 7) / 1000
        self.Why = randn(2, 7) / 1000
        #print(self.Why)
        self.bh = np.array([0]*7)
        self.by1 = 0
        self.by2 = 0

    def forward(self, inputs): #input is a 7 long list
        xh = [0] * 7
        for Hi in range(7):
            for j in range(len(inputs)):
                xh[Hi] += self.Wxh[Hi][j] * inputs[j]
            xh[Hi] = xh[Hi] + self.bh[Hi]
            xh[Hi] = np.tanh(xh[Hi])
            #print(self.Wxh[Hi][j])
        y1 = 0
        y2 = 0
        for Hi in range(7):
            y1 += self.Why[0][Hi] * xh[Hi]
            y2 += self.Why[1][Hi] * xh[Hi]
        y1 += self.by1
        y2 += self.by2
        return y1, y2

    def backprop(self, d_y, learn_rate = 2e-2, ):
        epochs = 1000
        





