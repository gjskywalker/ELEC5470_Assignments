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
def apply_lstm_model(data, labels, group, epochs=1500, batch_size=32):
    input_size = data.shape[2]
    model = LSTMModel(input_size, 50, 1)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    print(f'Total number of parameters: {total_params}')
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
        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    with open("lstm_losses"+str(group)+".pkl", "wb") as f:
        pickle.dump(losses, f)
    # Plot the training losses
    plt.figure()
    plt.plot(range(epochs), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss_lstm'+str(group)+'.pdf', dpi=800)

    return model

# Example usage
if __name__ == "__main__":
    '''
    In-sample Mean of predictions: 0.33297
    '''
    import os
    import pickle
    # Out-of-sample
    current_folder = os.path.dirname(os.getcwd())
    data_path = os.path.join(current_folder, "Prepare_Datasets/indicator_array.pkl")
    labels_path = os.path.join(current_folder, "Prepare_Datasets/labels.pkl")
    data = pickle.load(open(data_path, "rb"))
    # print(data)
    labels = pickle.load(open(labels_path, "rb"))
    # Apply the LSTM model
    model = apply_lstm_model(data, np.asarray(labels).reshape(2511,1), group=0)
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(data, dtype=torch.float32))
    '''
    Out-of-sample
    Total number of parameters: 12851
    Mean of predictions 0.3745983123779297 in 0
    Total number of parameters: 12851
    Mean of predictions 0.43026037216186525 in 1
    Total number of parameters: 12851
    Mean of predictions 0.36968929767608644 in 2
    Total number of parameters: 12851
    Mean of predictions 1.0852177365450188e-05 in 3
    Total number of parameters: 12851
    Mean of predictions 0.18390424251556398 in 4
    Total number of parameters: 12851
    Mean of predictions 1.775256387190893e-05 in 5
    
    import os
    import pickle
    # Out-of-sample
    current_folder = os.path.dirname(os.getcwd())
    data_path = os.path.join(current_folder, "Prepare_Datasets/out_of_sample_indicator_array.pkl")
    labels_path = os.path.join(current_folder, "Prepare_Datasets/out_of_sample_labels.pkl")
    data = pickle.load(open(data_path, "rb"))
    # print(data)
    labels = pickle.load(open(labels_path, "rb"))
    # Apply the LSTM model
    for i in range(6):
        model = apply_lstm_model(data[i]['train'], np.asarray(labels[i]['train']).reshape(400,1), group=i)
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(data[i]['test'], dtype=torch.float32))
        with open("lstm_predictions"+str(i)+".pkl", "wb") as f:
            pickle.dump(predictions, f)
        # Calculate the mean of predictions
        mean_prediction = predictions.sum().item() / predictions.shape[0]
        print(f'Mean of predictions {mean_prediction} in {i}') 
        print(f'Ground Truth: {np.mean(labels[i]["test"])}')
    '''
    