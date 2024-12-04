import pickle
import pandas as pd
import numpy as np

sp500_data = pd.read_csv('sp500_data.csv')
market_data = pd.read_csv('12_Industry_Portfolios_Daily.csv')
k = 30
indexes = dict()
for i, index in enumerate(sp500_data['Open']):
    indexes[i] = index

labels = list()
for i in range(k, len(indexes)-1):
    flag = 1
    for j in range(0, k):
        if indexes[i] <= indexes[i+j-k]:
            flag = 0
            break
    labels.append(flag)
print(len(labels))
with open("labels.pkl", "wb") as f:
    pickle.dump(np.asarray(labels).reshape(len(labels),1), f)

indicators = dict()
count = 0
for i in range(market_data.shape[0]):
    if int(market_data.iloc[i]['Date']) >= 20140101 and int(market_data.iloc[i]['Date']) <= 20240101:
        indicators[count] = np.asarray(market_data.iloc[i][1:].to_numpy(),dtype=np.float64)
        count += 1
with open("indicators.pkl", "wb") as f:
    pickle.dump(indicators, f)

# Create 3-D array for indicators
indicator_array = np.zeros((count - k - 1, k, len(indicators[0])), dtype=np.float64)
for i in range(k, count-1):
    for j in range(k):
        indicator_array[i-k, j, :] = indicators[i + j - k]
print(len(indicator_array))
with open("indicator_array.pkl", "wb") as f:
    pickle.dump(indicator_array, f)