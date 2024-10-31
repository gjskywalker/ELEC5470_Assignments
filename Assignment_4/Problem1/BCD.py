import typing as ty
import numpy as np  

MAX = 100

class BCD4LASSO:
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
        self.y = np.matmul(self.X, self.v) + 0.1 * np.random.randn(self.n, 1)
        self.lm = 0.2
    
    def sign(self, v: float) -> int:
        if v > 0:
            return 1
        elif v < 0 :
            return -1
        else :
            return 0
        
    def soft(self, v: float, c: float) -> float:
        return self.sign(v) * max((abs(v) - c/2), 0)
    
    def sum_font(self, vn: np.ndarray, n: int) -> np.ndarray:
        results = np.zeros([self.n, 1])
        for i in range(0, n):
            results += (self.X[:,i]*vn[i]).reshape(100, 1)
        print("sum font result: {0}".format(results))
        return results
    
    def sum_back(self, vn: np.ndarray, n: int) -> np.ndarray:
        results = np.zeros([self.n, 1])
        for i in range(n+1, self.p):
            results += (self.X[:,i]*vn[i]).reshape(100, 1)
        print("sum back result: {0}".format(results))
        return results
        
    def update_variable(self, vn: np.ndarray) -> np.ndarray:
        print("Initial Value : {0}".format(self.object_function(vn)))
        for i in range(0, self.p):
            y_t = self.y - self.sum_font(vn, i) - self.sum_back(vn, i)
            # print("Current y_t : {0}".format(y_t))
            print("Current Xi @ y_t : {0}".format(self.X[:,i].T @ y_t))
            print(np.linalg.norm(self.X[:,i], ord=2)**2)
            vn[i] = self.soft(self.X[:,i].T @ y_t, self.lm) / np.linalg.norm(self.X[:,i], ord=2)**2
            print("Current vn : {0}".format(vn))
            print("Current index : {0}, Current Function Value: {1}".format(i, self.object_function(vn)))
        return vn
    
    def object_function(self, vn: np.ndarray) -> float:
        return np.linalg.norm(self.y - np.matmul(self.X, vn), ord=2)**2 + self.lm * np.linalg.norm(vn, ord=1)
    
    def run(self) -> tuple:
        vn = self.v
        optimal_value = MAX
        initial_value = self.object_function(vn)
        while abs(initial_value - optimal_value) != 0:
            initial_value = optimal_value
            vn = self.update_variable(vn)
            optimal_value = self.object_function(vn)
        return vn, optimal_value
    
if __name__ == "__main__":
    BCD = BCD4LASSO()
    vn, optimal_value = BCD.run()
    print(vn)
    print(optimal_value)