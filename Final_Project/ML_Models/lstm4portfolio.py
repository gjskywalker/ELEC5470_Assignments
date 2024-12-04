import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the LSTM model
class LSTMModel(nn.Module):
    '''
    LSTMModel is a neural network model that uses Long Short-Term Memory (LSTM) layers
    to process sequential data and make binary classification predictions.

    Attributes:
        lstm (nn.LSTM): An LSTM layer that processes the input sequence.
        fc (nn.Linear): A fully connected layer that maps the LSTM output to the desired output size.
        sigmoid (nn.Sigmoid): A sigmoid activation function to convert the output to a probability.

    Methods:
        forward(x):
            Defines the forward pass of the model. Takes input tensor x and returns the output tensor.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Initializes the LSTMModel with the given input size, hidden size, and output size.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            output_size (int): The number of output features.
        '''
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        '''
        h_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(1, x.size(0), 50).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Function to apply the LSTM model
def apply_lstm_model(data, labels, epochs=2000, batch_size=32):
    input_size = data.shape[2]
    model = LSTMModel(input_size, 50, 1)
    criterion = nn.BCELoss()  # Use BCEWithLogitsLoss instead of BCELoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Plot the training losses
    plt.figure()
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss_lstm.pdf', dpi=800)

    return model

# Example usage
if __name__ == "__main__":
    import os
    import pickle
    current_folder = os.path.dirname(os.getcwd())
    data_path = os.path.join(current_folder, "Prepare_Datasets/indicator_array.pkl")
    labels_path = os.path.join(current_folder, "Prepare_Datasets/labels.pkl")
    data = pickle.load(open(data_path, "rb"))
    # print(data)
    labels = pickle.load(open(labels_path, "rb"))
    # Apply the LSTM model
    model = apply_lstm_model(data, labels)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data, dtype=torch.float32))
    print(predictions)
