import pickle
import pandas as pd
import numpy as np

class GenerateDataSet:
    def __init__(self, k):
        '''
        Initialize the GenerateDataSet class with the given time window size k.
        
        Parameters:
        k (int): The size of the time window for generating labels and indicators.
        
        Attributes:
        sp500_data (DataFrame): DataFrame containing S&P 500 data.
        market_data (DataFrame): DataFrame containing market data.
        k (int): The size of the time window.
        indexes (dict): Dictionary to store indexes from S&P 500 data.
        labels (list): List to store generated labels.
        indicators (dict): Dictionary to store market indicators.
        count (int): Counter for the number of valid market data entries.
        indicator_array (ndarray): Array to store the generated indicator array.
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
        '''
        Generate labels based on the S&P 500 data.
        
        The label is 1 if the current index is greater than all previous k indexes, otherwise 0.
        The labels are saved to a pickle file named "labels.pkl".
        '''
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
        '''
        Generate market indicators based on the market data.
        
        The indicators are stored in a dictionary with a counter as the key.
        Only data within the date range 20140101 to 20240101 is considered.
        The indicators are saved to a pickle file named "indicators.pkl".
        '''
        for i in range(self.market_data.shape[0]):
            if int(self.market_data.iloc[i]['Date']) >= 20140101 and int(self.market_data.iloc[i]['Date']) <= 20240101:
                self.indicators[self.count] = np.asarray(self.market_data.iloc[i][1:].to_numpy(), dtype=np.float64)
                self.count += 1
        with open("indicators.pkl", "wb") as f:
            pickle.dump(self.indicators, f)

    def generate_indicator_array(self):
        '''
        Generate the indicator array with the shape (len(self.labels), self.k, len(self.indicators[0])).
        
        The first dimension aligns with the labels, the second dimension aligns with the time window,
        and the third dimension aligns with the indicators feature size.
        The indicator array is saved to a pickle file named "indicator_array.pkl".
        '''
        self.indicator_array = np.zeros((len(self.labels), self.k, len(self.indicators[0])), dtype=np.float64)
        for i in range(self.k, self.count - 1):
            for j in range(self.k):
                self.indicator_array[i - self.k, j, :] = self.indicators[i + j - self.k]
        with open("indicator_array.pkl", "wb") as f:
            pickle.dump(self.indicator_array, f)

if __name__ == "__main__":
    '''
    Main function to create an instance of GenerateDataSet with a time window size of 5,
    and generate labels, indicators, and the indicator array.
    '''
    dataset = GenerateDataSet(k=5)
    dataset.generate_labels()
    dataset.generate_indicators()
    dataset.generate_indicator_array()