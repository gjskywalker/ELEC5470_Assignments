import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('lstm_losses0.pkl', 'rb') as f:
    lstm_losses0 = pickle.load(f)

with open('transformer_losses0.pkl', 'rb') as f:    
    transformer_losses0 = pickle.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

ax1.plot(lstm_losses0)
ax1.set_title('LSTM Losses', fontsize=12)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Losses', fontsize=12)
plt.subplots_adjust(wspace=0.3)
ax2.plot(transformer_losses0)
ax2.set_title('Transformer Losses', fontsize=12)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Losses', fontsize=12)
plt.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.15)

plt.savefig('lstm_vs_transformer_losses.pdf', dpi=800)

