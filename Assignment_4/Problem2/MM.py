import numpy as np

class MM4EPOS:
    def __init__(self, n: int):
        self.n = n
        np.random.seed(1)
        self.p = 10
        self.seed = 1
        np.random.seed(self.seed)
        self.v = np.zeros([self.p,1])
        self.v[2] = 1
        self.v[4] = 7
        self.v[9] = 3
        self.X = np.random.randn(self.n, self.p)
        self.y = np.matmul(self.X, self.v[0 : self.p]) + 0.1 * np.random.randn(self.n, 1)
        # self.y = np.random.randn(self.n, 1)
        
    def solution4epos(self, vn: np.ndarray) -> np.ndarray:
        sorted_indices = np.argsort(vn[:, 0])[::-1]  
        vn = vn[sorted_indices]
        rho = self.n
        for j in range(1, self.n+1):
            thre = vn[j-1] + (1/j) * (1 - np.sum(vn[0:j-1, 0]))
            if thre <= 0:
                rho = j - 1 
                break
        v = (1/rho) * (1 - np.sum(vn[0:rho, 0]))
        x = np.zeros([self.n, 1])
        for i in range(0, self.n):
            x[i] += max(vn[i]+v, 0)
        return x       
    
    def solution4epos_l1(self, vn: np.ndarray) -> np.ndarray:
        if np.linalg.norm(vn, ord=1) <= 1:
            return vn
        z = np.abs(vn)
        x = self.solution4epos(z)
        for i in range(0, self.n):
            if vn[i] > 0:
                x[i] = x[i]
            else:
                x[i] = -x[i]
        return x
    
    def objective_function(self, vn: np.ndarray, lm: float) -> float:
        return (1/2) * np.linalg.norm(vn - self.y, ord=2) + lm * np.sum((1/2)*np.log(1 + 2 * np.abs(vn))) 

    def solution4epos_sparse_l1(self, vn: np.ndarray, lmd: float) -> tuple:
        x = np.ones([self.n, 1]) / self.n
        old_value = self.objective_function(x, lmd)
        optimal_value = 0
        while old_value != optimal_value:
            old_value = optimal_value
            Lam = (2 * x**2 + np.abs(x)) / (lmd + (2 * x**2 + np.abs(x)))
            q = np.diag(Lam.reshape(self.n,)) @ vn
            x = self.solution4epos_l1(q)
            optimal_value = self.objective_function(x, lmd)
        return x, optimal_value
            
        

if __name__ == "__main__":
    mm = MM4EPOS(10)
    optimal_point, optimal_value = mm.solution4epos_sparse_l1(mm.y, 0.001)
    print("Optimal Point: {0}, Optimal Value: {1}, feasibility: {2}".format(optimal_point, optimal_value, np.sum(np.abs(optimal_point))))