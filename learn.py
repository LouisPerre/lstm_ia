import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/AAPL_stock_prediction_2")

data = pd.read_csv('AMZN.csv')

data = data[['Date', 'Close']]

device = 'cuda:0' if torch.cuda.is_available() else 'mps'

# Transform to a panda date type
data['Date'] = pd.to_datetime(data['Date'])
# Do a graph of the closing value with the date
plt.plot(data['Date'], data['Close'])

from copy import deepcopy as dc

# Take the dataframes and the number (look back window)
def prepare_dataframe_for_lstm(df, n_steps):
    # Deep copy of the dataframe
    df = dc(df)
    # Set the index to Date
    df.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 14
shifted_df = prepare_dataframe_for_lstm(data, lookback)

# Convert the data to numpy format
shifted_df_as_np = shifted_df.to_numpy()

class CustomMinMaxScaler:
    def __init__(self, min_val=-1, max_val=1, manual_max=None, increase_ratio=0.5):
        """
        Arguments:
        - min_val: La nouvelle valeur minimale pour la normalisation (par défaut -1).
        - max_val: La nouvelle valeur maximale pour la normalisation (par défaut 1).
        - manual_max: Si fourni, remplace le max calculé automatiquement du tableau.
        - increase_ratio: Si manual_max est None, augmente le max d'origine par ce ratio.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.manual_max = manual_max
        self.increase_ratio = increase_ratio
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, data):
        """Calcule les valeurs min et max nécessaires pour le scaling."""
        self.data_min_ = data.min()
        self.data_max_ = data.max()

        # Définir le max à utiliser
        if self.manual_max is not None:
            self.scaled_max_ = self.manual_max
        else:
            self.scaled_max_ = self.data_max_ * (1 + self.increase_ratio)
    
    def transform(self, data):
        """Applique la transformation min-max aux données."""
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        
        # Transformation à la plage [min_val, max_val]
        scaled = (data - self.data_min_) / (self.scaled_max_ - self.data_min_)
        return scaled * (self.max_val - self.min_val) + self.min_val

    def inverse_transform(self, scaled_data):
        """Inverse la transformation pour revenir à l'échelle d'origine."""
        if self.data_min_ is None or self.scaled_max_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        
        # Inverser la transformation
        original = (scaled_data - self.min_val) / (self.max_val - self.min_val)
        return original * (self.scaled_max_ - self.data_min_) + self.data_min_
    
    def fit_transform(self, data):
        """Combine fit et transform."""
        self.fit(data)
        return self.transform(data)

# Scaler on all the data
from sklearn.preprocessing import MinMaxScaler

scaler = CustomMinMaxScaler(min_val=-1, max_val=1, increase_ratio=0.5)
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
# Transform the numpy value from above to -1, 1
# scaler = MinMaxScaler(feature_range=(-1, 1))
# shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
# tableau_scaled

# X = All the rows above without the first column
X = shifted_df_as_np[:, 1:]
# Predicter, all the rows but just the first column
y = shifted_df_as_np[:, 0]

# Make a deep copy but flipping it, to have the oldest value first
X = dc(np.flip(X, axis=1))

# Splitting to train and test 95% as train and 5% as test
split_index = int(len(X) * 0.95)

# Up util the split index
X_train = X[:split_index]
# Split index onward
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Requirement for PyTorch LSTM to have an extra dimension at the end
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Wrap in pytorch tensor and make sure they are float
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# We use dataset instead of tensor
from torch.utils.data import Dataset

# Custom class to transform the tensor to a dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Wrap the dataset in dataloader to get the batches
from torch.utils.data import DataLoader

# Batch size of 16
batch_size = 16

# Train dataloader with our train_dataset, we shuffle everytime
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# The same but without shuffling
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Looping over our train_loader
for _, batch in enumerate(train_loader):
    # Get the x_batch and the y_batch to check the size
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

class LSTM(nn.Module):
    # Input size: number of features 1, hidden_size: However dimension we want in the middle could increase the overfitting,
    # num_stacked_layers: Stacked LSTM because they produce a sequence
    def __init__(self, input_size=1, hidden_size=64, num_stacked_layers=2, dropout_prob=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        # After LSTM you want a fully connected layer map of our hidden size to 1 because at the end you just need the final closing value
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Batch size from the input, dynamiccaly
        batch_size = x.size(0)
        # Gates, initalize the LSTM with default h0 and c0
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(x.device)
        # When you want to use the LSTM you pass those in the tuple, out = output, _ = updated tuple
        out, _ = self.lstm(x, (h0, c0))
        # The output has to pass in the fc layer
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)

def train_one_epoch():
    # Model to training mode
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    # Start to accumulate the running_loss
    running_loss = 0.0

    # Loop over the train loader and get the batch_index and the batch
    for batch_index, batch in enumerate(train_loader):
        # Get x_batch and y_batch from the batch and put it on the device we're using
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        # Getting the output from the model
        output = model(x_batch)
        # 'Tensor with one value' Getting the loss with the loss_function, comparing the output to the truth from the y_batch
        loss = loss_function(output, y_batch)
        # Incrementing the loss
        running_loss += loss.item()

        # Zero out the gradients
        optimizer.zero_grad()

        # Backward pass through the loss to calculate the gradient
        loss.backward()
        # Step in the direction of the gradient to upgrade our model
        optimizer.step()

        global_step = epoch * len(train_loader) + batch_index
        writer.add_scalar("Loss/Train", loss.item(), global_step)

        if batch_index % 100 == 99:  # print every 100 batches
            # Average loss across this 100 batches
            avg_loss_across_batches = running_loss / 100
            # writer.add_scalar('Train/Loss', avg_loss_across_batches, epoch * len(train_loader) + batch_index)
            print('Batch {0}, Loss: {1:.6f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

def validate_one_epoch():
    # Model to valuation mode
    model.train(False)
    running_loss = 0.0

    # Loop over the train loader and get the batch_index and the batch
    for batch_index, batch in enumerate(test_loader):
        # Get x_batch and y_batch from the batch and put it on the device we're using
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        # No model update so we don't need to calculate the gradient
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    writer.add_scalar("Loss/Validation", avg_loss_across_batches, epoch)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

def log_graph_to_tensorboard():
    dummy_input = torch.zeros((1, lookback, 1)).to(device)
    writer.add_graph(model, dummy_input)
log_graph_to_tensorboard()

learning_rate = 0.001
num_epochs = 10
# Regression value because we calculate a continuous value to minimize the mean squared error
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
sample_input = torch.randn(1, lookback, 1).to(device)
writer.add_graph(model, sample_input)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()
    # scheduler.step()
model_save_path = 'lstm_model.pth'
torch.save(model.state_dict(), model_save_path)

# No gradient for the prediction
with torch.no_grad():
    # Give the model the first 95% of the train data then give it to the cpu because numpy only use the cpu
    predicted = model(X_train.to(device)).to('cpu').numpy()


# Plot the y_train, real closing data
plt.plot(y_train, label='Actual Close')
# Plot the predicted closing value
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# Flatten to making sure it's one axis
train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
# Inverse transform to transform the -1 1 to real dollar
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])

# Same as above for the truth
dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])

plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# The same as above but for the predictions .detach() do the same as .no_grad()
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

def log_predictions_to_tensorboard(y_actual, y_predicted, title, step):
    fig, ax = plt.subplots()
    ax.plot(y_actual, label='Actual Close')
    ax.plot(y_predicted, label='Predicted Close')
    ax.set_xlabel('Day')
    ax.set_ylabel('Close')
    ax.legend()
    writer.add_figure(title, fig, global_step=step)

log_predictions_to_tensorboard(new_y_train, train_predictions, "Train Predictions", epoch)
log_predictions_to_tensorboard(new_y_test, test_predictions, "Test Predictions", epoch)

loaded_model = LSTM(1, 4, 1)
loaded_model.load_state_dict(torch.load(model_save_path))
loaded_model.to(device)
loaded_model.eval()

def predict_future_values(model, initial_sequence, n_future_steps, lookback):
    """
    Prédit les valeurs futures en utilisant le modèle LSTM.
    
    Arguments:
    - model: le modèle LSTM entraîné.
    - initial_sequence: la séquence initiale utilisée pour prédire les valeurs futures.
    - n_future_steps: le nombre de pas dans le futur à prédire.
    - lookback: la taille de la fenêtre glissante.
    
    Retourne:
    - Une liste des prédictions futures.
    """
    model.eval()
    predictions = []
    
    # Assurez-vous que la séquence initiale est un tenseur 3D [batch_size, lookback, features]
    input_seq = initial_sequence.reshape(1, lookback, 1).to(device)

    with torch.no_grad():
        for _ in range(n_future_steps):
            # Obtenir la prédiction pour l'étape suivante
            pred = model(input_seq)
            predictions.append(pred.item())
            
            # Mettre à jour la séquence en supprimant le plus ancien et en ajoutant la prédiction
            new_input = pred.reshape(1, 1, 1)  # Forme [batch_size, 1, features]
            input_seq = torch.cat((input_seq[:, 1:, :], new_input), dim=1)
    
    return predictions

initial_sequence = X_test[-1]

# Nombre de prédictions futures
n_future_steps = 7

# Prédire les valeurs futures
future_predictions = predict_future_values(loaded_model, initial_sequence, n_future_steps, lookback)

dummies = np.zeros((len(future_predictions), lookback + 1))
dummies[:, 0] = future_predictions
dummies = scaler.inverse_transform(dummies)
future_predictions = dummies[:, 0]

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Récupérer la dernière date du dataset
last_date = data['Date'].iloc[-1]

# Générer des dates futures à partir de la dernière date
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_future_steps, inclusive='left')

# Créer un DataFrame pour les résultats
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close': future_predictions
})

# Afficher le tableau des prédictions futures
print(future_df)

def log_table_to_tensorboard(df, title, step):
    """
    Loggue un tableau Pandas sous forme de texte dans TensorBoard.
    
    Arguments :
    - df : DataFrame à logger
    - title : Titre pour la section dans TensorBoard
    - step : Étape actuelle pour le suivi
    """
    # Convertir le DataFrame en texte formaté
    table_text = df.to_string(index=False)
    
    # Ajout au logger TensorBoard
    writer.add_text(title, f"```\n{table_text}\n```", global_step=step)

print(future_df)
log_table_to_tensorboard(future_df, 'Future Predictions', epoch)

# Fermer TensorBoard
writer.close()

