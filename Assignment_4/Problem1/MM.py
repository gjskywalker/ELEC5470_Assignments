import typing as ty
import numpy as np  
from BCD import BCD4LASSO
class MM4LASSO:
    def __init__(self):
        '''
        Initial all variables
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
        '''
        Soft an array elementwisely
        '''
        for i in range(0, self.p):
            vn[i] = BCD4LASSO.soft(self, vn[i], self.lm/c)
        return vn
     
    def update_variable(self, vn: np.ndarray) -> np.ndarray:
        '''
        Get the optimal point each time
        '''
        c = np.linalg.norm(np.matmul(self.X.T, self.X), ord=2) + 2 
        return self.soft_array(np.matmul(self.X.T, (self.y - np.matmul(self.X, vn)))/c + vn, c)
    
    def run(self) -> tuple:
        '''
        Do iterations and get the optimal value
        '''
        vn = self.v 
        old_value = BCD4LASSO.object_function(self, vn)
        optimal_value = 0
        print("Initial value: {0}".format(old_value))
        while old_value != optimal_value:
            old_value = optimal_value
            vn = self.update_variable(vn)
            optimal_value = BCD4LASSO.object_function(self, vn)
        return vn, BCD4LASSO.object_function(self, vn)
    

if __name__ == "__main__":
    mm = MM4LASSO()
    optimal_point, optimal_value = mm.run()
    print("Optimal_point: {0}, Optimal_value: {1}".format(optimal_point, optimal_value))
    '''
    Initial value: 3.3169241373838325
    Optimal_point: [[ 6.25935703e-03]
    [ 1.61680967e-02]
    [ 9.94260496e-01]
    [-8.67053507e-03]
    [ 6.99910430e+00]
    [ 6.72844119e-03]
    [ 9.42865503e-03]
    [ 5.17267789e-03]
    [ 1.66853950e-03]
    [ 2.98877662e+00]], Optimal_value: 3.2556020350993977
    '''
        
            
        
    
    