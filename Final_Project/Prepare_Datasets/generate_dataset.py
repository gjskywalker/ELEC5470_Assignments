import pickle
import pandas as pd
import numpy as np

sp500_data = pd.read_csv('sp500_data.csv')
market_data = pd.read_csv('12_Industry_Portfolios_Daily.csv')
k = 3
labels = dict()
for i, index in enumerate(sp500_data['Open']):
    labels[i] = index

for i in range(len(labels)-k):
    flag = 1
    for j in range(1, k+1):
        if labels[i] >= labels[i+j]:
            flag = 0
            break
    labels[i] = flag
print(labels)
# valid label length is 2512
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

indicators = dict()
count = 0
for i in range(market_data.shape[0]):
    if int(market_data.iloc[i]['Date']) >= 20140101 and int(market_data.iloc[i]['Date']) <= 20240101:
        indicators[count] = np.asarray(market_data.iloc[i][1:].to_numpy(),dtype=np.float64)
        count += 1
with open("indicators.pkl", "wb") as f:
    pickle.dump(indicators, f)  