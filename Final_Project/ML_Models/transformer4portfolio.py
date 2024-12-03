import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, nhead=2):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.transformer_encoder(x)
        out = self.fc(out)
        return out[:, -1, :]

# Function to apply the Transformer model
def apply_transformer_model(data, labels, epochs=400, batch_size=32):
    input_size = data.shape[2]
    model = TransformerModel(input_size, 50, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        print("outputs.shaoe: ", outputs.shape)
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
    # Apply the Transformer model
    model = apply_transformer_model(data, labels)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data, dtype=torch.float32))
    print(predictions)
