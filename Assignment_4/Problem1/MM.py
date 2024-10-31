import typing as ty
import numpy as np  
from BCD import BCD4LASSO
class MM4LASSO:
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
    
    def sign(self, v: float) -> int:
        if v > 0:
            return 1
        elif v < 0 :
            return -1
        else :
            return 0
        
    def soft_array(self, vn: np.ndarray, c: float) -> np.ndarray:
        for i in range(0, self.p):
            vn[i] = BCD4LASSO.soft(self, vn[i], self.lm/(2*c))
        return vn
     
    def update_variable(self, vn: np.ndarray) -> np.ndarray:
        c = np.linalg.norm(np.matmul(self.X.T, self.X), ord=2) + 2 
        print("L2(XTX):{0}".format(c))
        return self.soft_array(np.matmul(self.X.T, (self.y - np.matmul(self.X, vn)))/c + vn, c)
    
    def run(self) -> tuple:
        vn = self.v 
        old_value = BCD4LASSO.object_function(self, vn)
        optimal_value = 0
        cnt = 0
        print("Initial point: {0}, Initial value: {1}".format(vn, old_value))
        while cnt < 10:
            old_value = optimal_value
            vn = self.update_variable(vn)
            optimal_value = BCD4LASSO.object_function(self, vn)
            cnt += 1
            print("Iteration: {0}, vn: {1}, optimal_value: {2}".format(cnt, vn, optimal_value))
        return vn, BCD4LASSO.object_function(self, vn)
    

if __name__ == "__main__":
    mm = MM4LASSO()
    vn, value = mm.run()
    print("Optimal Point: {0}".format(vn))
    print("Optimal Value: {0}".format(value))    
        
            
        
    
    