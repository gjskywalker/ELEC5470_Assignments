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
        '''
        Compute the gradient
        '''
        beta = v[0:self.p, 0].reshape([10,1])
        t = v[self.p : 2*self.p, 0].reshape([10,1])
        df_beta = 2 * np.matmul(self.X.T, np.matmul(self.X, beta) - self.y) + (2/delta) * beta / (t**2 - beta**2)
        df_t = self.lm * np.ones([self.p, 1]) - (2/delta) * t / (t**2 - beta**2)
        return np.concatenate((df_beta, df_t))
    
    def calculate_hessian(self, delta:float, v: np.ndarray) -> np.ndarray:
        '''
        Compute the hessian matrix
        '''
        H = np.zeros([2*self.p, 2*self.p])
        X_square = 2 * np.matmul(self.X.T, self.X)
        p1 = np.zeros(self.p)
        p2 = np.zeros(self.p)
        beta = v[0:self.p, 0]
        t = v[self.p : 2*self.p, 0]
        p1 = (2/delta) * (beta**2 + t**2) / (beta**2 - t**2)**2
        p2 = -(4/delta) * (beta * t) / (t**2 - beta**2)**2
        P1 = np.diag(p1.reshape(self.p,))
        P2 = np.diag(p2.reshape(self.p,))
        H = np.vstack((np.hstack((X_square, P2)),np.hstack((P2, P1))))
        return H
    
    def newton_method(self, delta: float, vn: np.ndarray, stepsize: float, tolerance: float) -> np.ndarray:
        '''
        Do newton, but during running the code, there is one thing weird that det(H) may equal to zero and it seems related to numpy bugs (overflow encountered). So I add 
        one extra condition to get rid of it.
        '''
        g = self.calculate_gradient(delta, vn)
        H = self.calculate_hessian(delta, vn)
        if np.linalg.det(H) == 0:
            return vn
        cnt = 0
        max_iteration = 100
        while (np.matmul(g.T, np.matmul(np.linalg.inv(H), g)) / 2 > tolerance) and (cnt < max_iteration):
            vn = vn - stepsize * np.matmul(np.linalg.inv(H), g)
            g = self.calculate_gradient(delta, vn)
            H = self.calculate_hessian(delta, vn)
            if np.linalg.det(H) == 0:
                return vn
            cnt += 1
        return vn
        
    def run(self, delta: float, u: float, tolerance: float, stepsize: float) -> tuple:
        vn = self.v
        max_iteration = 10
        cnt = 0
        print("Initial_value: {0}".format(np.linalg.norm(self.y - np.matmul(self.X, vn[0 : self.p]), ord=2)**2 + self.lm * np.linalg.norm(vn[0 : self.p], ord=1)))
        while (self.p * 2 / delta > tolerance) and (cnt < max_iteration):
            vn = self.newton_method(delta, vn, stepsize, tolerance)
            delta *= u 
            cnt += 1
        optimal_value = np.linalg.norm(self.y - np.matmul(self.X, vn[0 : self.p]), ord=2)**2 + self.lm * np.linalg.norm(vn[0 : self.p], ord=1)
        return vn, optimal_value
        
if __name__ == "__main__":
    BM = BM4LASSO()
    vn, optimal_value = BM.run(0.1, 100, 1e-6, 1e-2)
    print("Optimal_point: {0}, Optimal_value: {1}".format(vn[0:BM.p], optimal_value))
    '''
    Initial_value: 3.3169241373838325
    Optimal_point: [[ 6.98879958e-03]
    [ 1.60315249e-02]
    [ 9.97248848e-01]
    [-8.41534863e-03]
    [ 7.00068738e+00]
    [ 6.53871108e-03]
    [ 9.22284839e-03]
    [ 5.37780500e-03]
    [ 3.22300670e-03]
    [ 2.99295109e+00]], Optimal_value: 3.258439443916438
    '''

    
        
        