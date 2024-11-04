import typing as ty
import numpy as np  
from BCD import BCD4LASSO
class SCA4LASSO:
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
        self.bcd = BCD4LASSO()
        
    def update_variable(self, vn: np.ndarray, tau: float) -> np.ndarray:
        '''
        Get the optimal point each time
        '''
        for i in range(0, self.p):
            y_t = self.y - self.bcd.sum_back(vn, i) - self.bcd.sum_font(vn, i)
            vn[i] = self.bcd.soft(np.matmul(self.X[:,i].T, y_t) + tau*vn[i], self.lm) / (tau + np.linalg.norm(self.X[:,i], ord=2)**2)
        return vn
    
    def run(self, tau:float, gamma: float) -> tuple:
        '''
        Do iterations and get the optimal value
        '''
        vn = self.v 
        old_value = self.bcd.object_function(vn)
        optimal_value = 0
        print("Initial value: {0}".format(old_value))
        while old_value != optimal_value:
            old_value = optimal_value
            vn = vn + gamma * (self.update_variable(vn, tau) - vn)
            optimal_value = self.bcd.object_function(vn)
        return vn, self.bcd.object_function(vn)
    
if __name__ == "__main__":
    sca = SCA4LASSO()
    optimal_point, optimal_value = sca.run(tau=1, gamma=0.3)
    print("Optimal_point: {0}, Optimal_value: {1}".format(optimal_point, optimal_value))
    '''
    Initial value: 3.3169241373838325
    Optimal_point: [[ 6.25935754e-03]
    [ 1.61680992e-02]
    [ 9.94260495e-01]
    [-8.67053420e-03]
    [ 6.99910430e+00]
    [ 6.72844241e-03]
    [ 9.42865592e-03]
    [ 5.17267857e-03]
    [ 1.66853983e-03]
    [ 2.98877662e+00]], Optimal_value: 3.255602035099397
    '''
    
    