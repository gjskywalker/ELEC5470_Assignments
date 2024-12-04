import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define the Transformer model
class TransformerModel(nn.Module):
    '''
    TransformerModel is a neural network model based on the Transformer architecture.
    It consists of an encoder layer, a transformer encoder, a fully connected layer, and a sigmoid activation function.
    
    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units in the feedforward layers.
        output_size (int): The number of output features.
        num_layers (int, optional): The number of encoder layers in the transformer. Default is 1.
        nhead (int, optional): The number of heads in the multiheadattention models. Default is 2.
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, nhead=2):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        '''
        out = self.transformer_encoder(x)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out[:, -1, :]

# Function to apply the Transformer model
def apply_transformer_model(data, labels, epochs=10000, batch_size=32):
    '''
    Trains the Transformer model on the provided data and labels.
    
    Args:
        data (ndarray): Input data of shape (num_samples, sequence_length, num_features).
        labels (ndarray): Target labels of shape (num_samples, 1).
        epochs (int, optional): Number of training epochs. Default is 10000.
        batch_size (int, optional): Batch size for training. Default is 32.
    
    Returns:
        TransformerModel: The trained Transformer model.
    '''
    input_size = data.shape[2]
    model = TransformerModel(input_size, 50, 1)
    criterion = nn.MSELoss()
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
    plt.savefig('training_loss_transformer.pdf', dpi=800)

    return model

# Example usage
if __name__ == "__main__":
    '''
    Loads the data and labels from pickle files, trains the Transformer model, and prints the predictions.
    '''
    import os
    import pickle
    current_folder = os.path.dirname(os.getcwd())
    data_path = os.path.join(current_folder, "Prepare_Datasets/indicator_array.pkl")
    labels_path = os.path.join(current_folder, "Prepare_Datasets/labels.pkl")
    data = pickle.load(open(data_path, "rb"))
    # print(data)
    labels = pickle.load(open(labels_path, "rb"))
    # Apply the LSTM model
    # Apply the Transformer model
    model = apply_transformer_model(data, labels)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data, dtype=torch.float32))
    print(predictions)
