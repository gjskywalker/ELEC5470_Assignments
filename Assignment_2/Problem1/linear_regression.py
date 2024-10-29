import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class linear_regression:
    def __init__(self, dataset: str, picname: str):
        self.x: np.ndarray
        self.y: np.ndarray
        self.a: float
        self.b: float
        self.e: float
        self.a_des = 3.0
        self.b_des = 4.0
        self.dataset = dataset
        self.picname = picname

    def read_dataset(self) -> None:
        df = pd.read_csv(self.dataset, index_col=0)
        df.columns = ["x", "y"]
        self.x = np.asarray(df["x"])
        self.y = np.asarray(df["y"])
        return None

    def square_penalty(self) -> None:
        a = cp.Variable(1)
        b = cp.Variable(1)
        objective = cp.Minimize(cp.sum_squares(a * self.x + b - self.y))
        prob = cp.Problem(objective)
        result = prob.solve()
        self.a = a.value
        self.b = b.value
        des_value = np.array([self.a_des, self.b_des]).reshape(
            2,
        )
        est_value = np.array([self.a, self.b]).reshape(
            2,
        )
        self.e = np.linalg.norm(est_value - des_value)
        return None

    def norm1_penalty(self) -> None:
        a = cp.Variable(1)
        b = cp.Variable(1)
        objective = cp.Minimize(cp.sum(cp.abs(a * self.x + b - self.y)))
        prob = cp.Problem(objective)
        result = prob.solve()
        self.a = a.value
        self.b = b.value
        des_value = np.array([self.a_des, self.b_des]).reshape(
            2,
        )
        est_value = np.array([self.a, self.b]).reshape(
            2,
        )
        self.e = np.linalg.norm(est_value - des_value)
        return None

    def plot(self) -> None:
        plt.scatter(self.x, self.y, label="Data points", color="blue")
        z = self.a * self.x + self.b
        plt.plot(
            self.x,
            z,
            color="red",
            label=f"Curve: z = {self.a}x + {self.b} \nEstimation error: {self.e}",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Linear Regression with {}".format(self.picname))
        plt.savefig("{}.png".format(self.picname), dpi=600)
        plt.close()


if __name__ == "__main__":
    lire_norm_1 = linear_regression(dataset="dataset1.csv", picname="norm1")
    lire_norm_1.read_dataset()
    lire_norm_1.norm1_penalty()
    lire_norm_1.plot()

    lire_sqa_1 = linear_regression(dataset="dataset1.csv", picname="sqa")
    lire_sqa_1.read_dataset()
    lire_sqa_1.square_penalty()
    lire_sqa_1.plot()
