import typing as ty
import numpy as np  

MAX = 100

class BCD4LASSO:
    def __init__(self):
        '''
        Initial All variables
        '''
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
        '''
        x = sign(x) * max(|x| - c/2, 0)
        '''
        return self.sign(v) * max((abs(v) - c/2), 0)
    
    def sum_font(self, vn: np.ndarray, n: int) -> np.ndarray:
        '''
        sum_{0~i} x_i*vn[i]
        '''
        results = np.zeros([self.n, 1])
        for i in range(0, n):
            results += (self.X[:,i]*vn[i]).reshape(100, 1)
        return results
    
    def sum_back(self, vn: np.ndarray, n: int) -> np.ndarray:
        '''
        sum_{i+1~n} x_i*vn[i]
        '''
        results = np.zeros([self.n, 1])
        for i in range(n+1, self.p):
            results += (self.X[:,i]*vn[i]).reshape(100, 1)
        return results
        
    def update_variable(self, vn: np.ndarray) -> np.ndarray:
        '''
        Get the optimal point each time
        '''
        for i in range(0, self.p):
            y_t = self.y - self.sum_font(vn, i) - self.sum_back(vn, i)
            vn[i] = self.soft(self.X[:,i].T @ y_t, self.lm) / np.linalg.norm(self.X[:,i], ord=2)**2
        return vn
    
    def object_function(self, vn: np.ndarray) -> float:
        return np.linalg.norm(self.y - np.matmul(self.X, vn), ord=2)**2 + self.lm * np.linalg.norm(vn, ord=1)
    
    def run(self) -> tuple:
        '''
        Do iterations and get the optimal value
        '''
        vn = self.v
        optimal_value = MAX
        initial_value = self.object_function(vn)
        print("Initial Value : {0}".format(self.object_function(vn)))
        while abs(initial_value - optimal_value) != 0:
            initial_value = optimal_value
            vn = self.update_variable(vn)
            optimal_value = self.object_function(vn)
        return vn, optimal_value
    
if __name__ == "__main__":
    BCD = BCD4LASSO()
    vn, optimal_value = BCD.run()
    print("Optimal_point: {0}, Optimal_value: {1}".format(vn, optimal_value))
    '''
    Initial Value : 3.3169241373838325
    Optimal_point: [[ 6.25935767e-03]
    [ 1.61680991e-02]
    [ 9.94260495e-01]
    [-8.67053413e-03]
    [ 6.99910430e+00]
    [ 6.72844242e-03]
    [ 9.42865582e-03]
    [ 5.17267844e-03]
    [ 1.66853995e-03]
    [ 2.98877662e+00]], Optimal_value: 3.255602035099394
    '''