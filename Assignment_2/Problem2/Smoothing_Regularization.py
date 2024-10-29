import numpy as np
import pandas as pd
import cvxpy as cp  
import math 
import matplotlib.pyplot as plt

class Smoothing_regularization:
    def __init__(self, filename):
        self.dataset = pd.read_csv(filename)
        self.y_des = np.asarray(self.dataset.iloc[:, 0])
        self.N = 400
        self.u1 : np.ndarray
        self.u2 : np.ndarray
        self.uinf : np.ndarray
        self.u : np.ndarray
        self.cm : np.ndarray
    
    def plot1(self, x: np.ndarray, u: np.ndarray, y: np.ndarray, picname : str) -> None:
        # Create a figure with size (2, 1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Subplot (0, 0): Scatter plot of (x, y)
        axs[0].plot(x, u, color='blue', label='u')
        axs[0].set_title('input u')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('u')
        axs[0].legend()

        # Subplot (0, 1): Two scatter plots of (x, y1) and (x, y2)
        axs[1].plot(x, self.cm @ u, color='red', label='optimized output')
        axs[1].plot(x, y, color='green', label='desired output')
        axs[1].set_title('Comparision between optimized output and desired output')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig("{}.png".format(picname), dpi = 600)
    
    def plot2(self, x: np.ndarray, u: np.ndarray, y: np.ndarray, delta : float, eta : float) -> None:
        # Create a figure with size (2, 1)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Subplot (0, 0): Scatter plot of (x, y)
        axs[0].plot(x, u, color='blue', label='u')
        axs[0].set_title('input u')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('u')
        axs[0].legend()

        # Subplot (0, 1): Two scatter plots of (x, y1) and (x, y2)
        axs[1].plot(x, self.cm @ u, color='red', label='optimized output')
        axs[1].plot(x, y, color='green', label='desired output')
        axs[1].set_title(r'Smoothing Regularization with $\delta$:{} $\eta$:{}'.format(delta, eta))
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].legend()

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"delta_{delta:.2f}eta_{eta:.2f}.png", dpi = 600)

    def h(self, t : int) -> float:
        return (1/9) * math.pow(0.9, t) * (1 - 0.4 * math.sin(3*t))
    
    def build_matrix(self) -> None:
        cm = np.zeros((self.N, self.N))
        for i in range(0, self.N):
            for j in range(0, i+1):
                cm[i, j] = self.h(i-j)
        self.cm = cm
    
    def norm1_penalty(self) -> None:
        self.build_matrix()
        u1 = cp.Variable(self.N)
        objective = cp.Minimize(cp.norm1(self.cm @ u1 - self.y_des))
        prob = cp.Problem(objective)
        prob.solve()
        self.u1 = np.asarray(u1.value)
        self.plot1(np.arange(self.N), self.u1, self.y_des, "norm1")
    
    def norm2_penalty(self) -> None:
        self.build_matrix()
        u2 = cp.Variable(self.N)
        objective = cp.Minimize(cp.norm1(self.cm @ u2 - self.y_des))
        prob = cp.Problem(objective)
        prob.solve()
        self.u2 = np.asarray(u2.value)
        self.plot1(np.arange(self.N), self.u2, self.y_des, "norm2")

    def norminf_penalty(self) -> None:
        self.build_matrix()
        uinf = cp.Variable(self.N)
        objective = cp.Minimize(cp.norm1(self.cm @ uinf - self.y_des))
        prob = cp.Problem(objective)
        prob.solve()
        self.uinf = np.asarray(uinf.value)
        self.plot1(np.arange(self.N), self.uinf, self.y_des, "norminf")

    def tradeoff_penalty(self, delta: float, eta: float) -> None:
        self.build_matrix()
        u = cp.Variable(self.N)
        objective = cp.Minimize(eta * cp.square(cp.norm2(u)) + cp.norm2(self.cm @ u - self.y_des) + delta * cp.sum_squares(u[1:] - u[:-1]))
        prob = cp.Problem(objective)
        prob.solve()
        self.u = np.asarray(u.value)
        self.plot2(np.arange(self.N), self.u, self.y_des, delta, eta)


if __name__ == "__main__":
    sr = Smoothing_regularization("dataset3.csv")
    sr.norminf_penalty()






















if __name__ == "__main__":
    sr = Smoothing_regularization("dataset3.csv")
    