import numpy as np
from numpy.random import randn


class Network:

    def __init__(self):
        self.Wxh = randn(7, 7) / np.sqrt(7)
        self.Why = randn(2, 7) / np.sqrt(7)

        self.bh = np.zeros(7)
        self.by = np.zeros(2)

        self.last_hs = np.zeros(7)
        self.last_inputs = np.zeros(7)

        self.last_out = np.zeros(2)

    def forward(self, inputs):

        self.last_inputs = np.array(inputs, dtype=float)

        xh = np.zeros(7, dtype=float)

        for i in range(7):
            for j in range(7):
                xh[i] += self.Wxh[i][j] * inputs[j]

            xh[i] += self.bh[i]
            xh[i] = np.tanh(xh[i])

        self.last_hs = xh.copy()

        y = np.zeros(2)

        for i in range(2):
            for j in range(7):
                y[i] += self.Why[i][j] * xh[j]

            y[i] += self.by[i]

        self.last_out = y.copy()
        return y


    def backprop(self, d_y, learn_rate=0.0005):

        self.by -= learn_rate * d_y

        d_why = np.zeros_like(self.Why)

        for i in range(2):
            for j in range(7):
                d_why[i][j] = d_y[i] * self.last_hs[j]

        self.Why -= learn_rate * d_why

        dh = np.zeros(7)

        for j in range(7):
            for i in range(2):
                dh[j] += self.Why[i][j] * d_y[i]

            dh[j] *= (1 - self.last_hs[j] ** 2)

        d_wxh = np.zeros_like(self.Wxh)
        d_bh = np.zeros_like(self.bh)

        for j in range(7):
            d_bh[j] = dh[j]

            for k in range(7):
                d_wxh[j][k] = dh[j] * self.last_inputs[k]

        self.Wxh -= learn_rate * d_wxh
        self.bh -= learn_rate * d_bh

