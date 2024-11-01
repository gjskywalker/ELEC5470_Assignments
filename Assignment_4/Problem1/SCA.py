import typing as ty
import numpy as np  
from BCD import BCD4LASSO
class SCA4LASSO:
    def __init__(self):
        self.n = 100
        self.p = 10
        self.seed = 1
        np.random.seed(self.seed)
        self.v = np.zeros([self.p,1])
        self.v[2] = 1
        self.v[4] = 7
        self.v[9] = 3
        self.X = np.random.randn(self.n, self.p)
        self.y = np.matmul(self.X, self.v[0 : self.p]) + 0.1 * np.random.randn(self.n, 1)
        self.lm = 0.2
        self.bcd = BCD4LASSO()
        
    def update_variable(self, vn: np.ndarray, tau: float) -> np.ndarray:
        for i in range(0, self.p):
            y_t = self.y - self.bcd.sum_back(vn, i) - self.bcd.sum_font(vn, i)
            vn[i] = self.bcd.soft(np.matmul(self.X[:,i].T, y_t) + tau*vn[i], self.lm) / (tau + np.linalg.norm(self.X[:,i], ord=2)**2)
        return vn
    
    def run(self, tau:float, gamma: float) -> tuple:
        vn = self.v 
        old_value = self.bcd.object_function(vn)
        optimal_value = 0
        cnt = 0
        print("Initial point: {0}, Initial value: {1}".format(vn, old_value))
        while cnt < 10:
            old_value = optimal_value
            vn = vn + gamma * (self.update_variable(vn, tau) - vn)
            optimal_value = self.bcd.object_function(vn)
            cnt += 1
            print("Iteration: {0}, Current Point: {1}, Current Value: {2}".format(cnt, vn, optimal_value))
        return vn, self.bcd.object_function(vn)
    
if __name__ == "__main__":
    sca = SCA4LASSO()
    optimal_point, optimal_value = sca.run(tau=0.1, gamma=0.3)
    print("Optimal_point: {0}, Optimal_value: {1}".format(optimal_point, optimal_value))
    
    