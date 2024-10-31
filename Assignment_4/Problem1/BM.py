import typing as ty
import numpy as np  

class BM4LASSO:
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
        
    def calculate_gradient(self, delta: float, v: np.ndarray) -> np.ndarray:
        df_beta = 2 * delta * np.matmul(self.X.T, np.matmul(self.X, v[0 : self.p]) - self.y)
        df_t = delta * self.lm * np.ones([self.p, 1])
        for i in range(0, self.p):
            df_beta[i] += 2*v[i] / (v[self.p+i]**2 - v[i]**2)
            df_t -= 2*v[self.p+i] / (v[self.p+i]**2 - v[i]**2)
        # print(np.concatenate((df_beta, df_t)))
        return np.concatenate((df_beta, df_t))
    
    def calculate_hessian(self, delta:float, v: np.ndarray) -> np.ndarray:
        H = np.zeros([2*self.p, 2*self.p])
        X_square = 2 * delta * np.matmul(self.X.T, self.X)
        p1 = np.zeros(self.p)
        p2 = np.zeros(self.p)
        for i in range(0, self.p):
            p1[i] += 2 * (v[i]**2 + v[self.p+i]**2) / (v[self.p+i]**2 - v[i]**2)**2
            p2[i] += -4 * (v[i]*v[self.p+i]) / (v[i]**2 - v[self.p+i]**2)**2
        P1 = np.diag(p1)
        P2 = np.diag(p2)
        for i in range(0, self.p):
            for j in range(0, self.p):
                H[i][j] += X_square[i][j] + P1[i][j]
                H[i+self.p][j] += P2[i][j]
                H[i][j+self.p] += P2[i][j]
                H[i+self.p][j+self.p] += P1[i][j]
        print(X_square)
        print("=================")
        print(P1)
        print("=================")
        print(P2)
        print("=================")
        print(H)
        print("=================")
        return H
    
    def newton_method(self, delta: float, vn: np.ndarray, stepsize: float, tolerance: float) -> np.ndarray:
        g = self.calculate_gradient(delta, vn)
        H = self.calculate_hessian(delta, vn)
        cnt = 0
        while np.matmul(g.T, np.matmul(np.linalg.inv(H), g)) > tolerance * 1e5:
        # while cnt < 10:
            vn = vn - stepsize * np.matmul(np.linalg.inv(H), g)
            # cnt += 1
        return vn
        
    def run(self, delta: float, u: float, tolerance: float, stepsize: float) -> tuple:
        vn = self.v
        while self.p * 2 / delta > tolerance:
            vn = self.newton_method(delta, vn, stepsize, tolerance)
            # print(vn[0 : self.p])
            delta *= u 
            # print(delta)      
        # print(vn[0 : self.p]) 
        optimal_value = np.linalg.norm(self.y - np.matmul(self.X, vn[0 : self.p]), ord=2)**2 + self.lm * np.linalg.norm(vn[0 : self.p], ord=1)
        return vn, optimal_value
        
if __name__ == "__main__":
    BM = BM4LASSO()
    # BM.calculate_hessian(10, BM.v)
    vn, optimal_value = BM.run(0.1, 10, 1e-9, 1e-2)
    print(vn[0: BM.p])
    print(optimal_value)

    
        
        