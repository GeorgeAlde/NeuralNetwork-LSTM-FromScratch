import numpy as np
from numpy.random import randn


class Network:

    def __init__(self):
        self.Wxh = randn(7, 7) / 1000
        self.Why = randn(2, 7) / 1000
        #print(self.Wxh[1])
        self.bh = [0,0,0,0,0,0,0]
        self.by = [0,0]
        self.last_hs = []
        self.last_inputs = []

    def forward(self, inputs): #input is a 7 long list
        xh = [0] * 7
        self.last_hs = []
        self.last_inputs = []
        for i in range(7):
            for Hi in range(7):
                xh[i] += self.Wxh[i][Hi] * inputs[Hi]
            xh[i] = xh[i] + self.bh[i]
            xh[i] = np.tanh(xh[i])
            self.last_hs.append(xh[i])
            self.last_inputs = inputs
            #print(self.Wxh[Hi][j])
        y0 = 0
        y1 = 0
        for Hi in range(7):
            y0 += self.Why[0][Hi] * xh[Hi]
            y1 += self.Why[1][Hi] * xh[Hi]
        y0 += self.by[0]
        y1 += self.by[1]
        return y0, y1

    def backprop(self, d_y, y_index, gradient_num = 1000, learn_rate = 2e-2):
        d_by = d_y
        d_by = np.clip(d_by, -1, 1)
        for i in range(gradient_num):
            for j in range(7):
                d_why = d_y * self.last_hs[j]
                d_why = np.clip(d_why, -1, 1)
                d_bh = d_y * self.Why[y_index,j]
                d_bh = np.clip(d_bh, -1, 1)
                for k in range(7):
                    d_wxh = d_y * self.Why[y_index, j] * self.last_inputs[k]
                    d_wxh = np.clip(d_wxh, -1 , 1)

                    step = d_wxh * learn_rate
                    self.Wxh[k,j] -= step


                step = d_why * learn_rate
                self.Why[y_index, j] -= step
                #print(self.Why)
                step = d_bh * learn_rate
                self.bh[j] -= step

        step = d_by * learn_rate
        self.by[y_index] -= step







        





