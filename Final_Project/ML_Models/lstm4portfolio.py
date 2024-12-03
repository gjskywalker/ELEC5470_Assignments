import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Function to apply the LSTM model
def apply_lstm_model(data, labels, epochs=400, batch_size=32):
    input_size = data.shape[2]
    model = LSTMModel(input_size, 50, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# Example usage
if __name__ == "__main__":
    # Generate dummy data
    data = np.random.rand(50, 300, 4)  # 100 samples, 10 time steps, 1 feature
    labels = np.random.rand(50, 1)    # 100 labels
    print("Labels: ", labels)
    # Apply the LSTM model
    model = apply_lstm_model(data, labels)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data, dtype=torch.float32))
    print(predictions)
