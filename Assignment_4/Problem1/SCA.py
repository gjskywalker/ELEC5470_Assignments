import typing as ty
import numpy as np  

class SCA4LASSO:
    def __init__(self):
        self.n = 100
        self.p = 10
        self.seed = 1
        np.random.seed(self.seed)
        self.v = np.zeros([2*self.p,1])
        self.v[2] = 1
        self.v[4] = 7
        self.v[9] = 3
        for i in range(0, self.p):
            self.v[i+self.p] = self.v[i] + 1
        self.X = np.random.randn(self.n, self.p)
        self.y = np.matmul(self.X, self.v[0 : self.p]) + 0.1 * np.random.randn(self.n, 1)
        self.lm = 0.2