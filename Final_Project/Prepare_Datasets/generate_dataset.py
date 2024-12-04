import pickle
import pandas as pd
import numpy as np
class GenerateDataSet:
    def __init__(self, k):
        '''
        ML_Models/transformer4portfolio.py
        '''
        self.sp500_data = pd.read_csv('sp500_data.csv')
        self.market_data = pd.read_csv('12_Industry_Portfolios_Daily.csv')
        self.k = k 
        self.indexes = dict()
        self.labels = list()
        self.indicators = dict()
        self.count = 0
        self.indicator_array = None

    def generate_labels(self):
        for i, index in enumerate(self.sp500_data['Open']):
            self.indexes[i] = index

        for i in range(self.k, len(self.indexes)):
            flag = 1
            for j in range(0, self.k):
                if self.indexes[i] <= self.indexes[i + j - self.k]:
                    flag = 0
                    break
            self.labels.append(flag)
        with open("labels.pkl", "wb") as f:
            pickle.dump(np.asarray(self.labels).reshape(len(self.labels), 1), f)

    def generate_indicators(self):
        for i in range(self.market_data.shape[0]):
            if int(self.market_data.iloc[i]['Date']) >= 20140101 and int(self.market_data.iloc[i]['Date']) <= 20240101:
                self.indicators[self.count] = np.asarray(self.market_data.iloc[i][1:].to_numpy(), dtype=np.float64)
                self.count += 1
        with open("indicators.pkl", "wb") as f:
            pickle.dump(self.indicators, f)

    def generate_indicator_array(self):
        '''
        the shape of the indicator array is (len(self.labels), self.k, len(self.indicators[0])), 
        the first dimension aligns with the labels, 
        the second dimension aligns with the time window, 
        the third dimension aligns with the indicators feature size
        '''
        self.indicator_array = np.zeros((len(self.labels), self.k, len(self.indicators[0])), dtype=np.float64)
        for i in range(self.k, self.count - 1):
            for j in range(self.k):
                self.indicator_array[i - self.k, j, :] = self.indicators[i + j - self.k]
        with open("indicator_array.pkl", "wb") as f:
            pickle.dump(self.indicator_array, f)

if __name__ == "__main__":
    dataset = GenerateDataSet(k=5)
    dataset.generate_labels()
    dataset.generate_indicators()
    dataset.generate_indicator_array()