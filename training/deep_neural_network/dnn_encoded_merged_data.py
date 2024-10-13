# %%
# Import
import numpy as np 
import pandas as pd
import torch as tc
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from metric import MAPE, RMSE, MAE, RRSE, RSE, MSE, MSLE, RMSLE


# %%
err_dict = {'MAPE': MAPE, 'RMSE': RMSE, 'MAE': MAE, 'MSE': MSE, 'RRSE': RRSE, 'RSE': RSE, 'MSLE': MSLE, 'RMSLE': RMSLE}

# %%
class DNN(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=1, dtype=tc.float64):
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]

        self.seq = nn.Sequential()
        self.seq.append(nn.Linear(input_size, hidden_size[0], dtype=dtype))
        self.seq.append(nn.ReLU())
        for i in range(len(hidden_size)):
            if i == len(hidden_size) - 1:
                self.seq.append(nn.Linear(hidden_size[i], output_size, dtype=dtype))
            else:
                self.seq.append(nn.Linear(hidden_size[i], hidden_size[i + 1], dtype=dtype))
                self.seq.append(nn.ReLU())

    def forward(self, input):
        return self.seq(input)

# %%
def train_model(train_inputs, train_labels, valid_inputs, valid_labels, model, optimizer, n_epochs=10, batch_size=64, loss_name='MSE', metric_names=['MAPE']):
    
    loss_fn = err_dict[loss_name]
    log_dict = {f'{loss_name}_loss': {'train': np.zeros(n_epochs), 'valid': np.zeros(n_epochs)}}

    for metric_name in metric_names:
        log_dict[f'{metric_name}_metric'] = {'train': np.zeros(n_epochs), 'valid': np.zeros(n_epochs)}

    train_data_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=batch_size)
    for i in range(n_epochs):
        model.train()
        for data in train_data_loader:
            inputs, labels = data
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with tc.no_grad():
            train_preds = model(train_inputs)
            log_dict[f'{loss_name}_loss']['train'][i] = loss_fn(train_preds, train_labels)
            for metric_name in metric_names:
                log_dict[f'{metric_name}_metric']['train'][i] = err_dict[metric_name](train_preds, train_labels)

            valid_preds = model(valid_inputs)
            log_dict[f'{loss_name}_loss']['valid'][i] = loss_fn(valid_preds, valid_labels)
            for metric_name in metric_names:
                log_dict[f'{metric_name}_metric']['valid'][i] = err_dict[metric_name](valid_preds, valid_labels)

            log_info = f'Epoch {i+1}/{n_epochs} - '
            for log_name, log_values in log_dict.items():
                train_log_values = log_values['train']
                valid_log_values = log_values['valid']
                log_info += f'train_{log_name}: {train_log_values[i]:.3f} - valid_{log_name}: {valid_log_values[i]:.3f} - '
            print(log_info)

    return log_dict


# %%
orig_df = pd.read_csv('expand_df.csv', header=0, keep_default_na=False)

# %%
drop_df = orig_df.drop(columns=['Unnamed: 0', 'ID', '路名', '備註'])

# %%
enc_df = pd.get_dummies(drop_df, columns=['縣市', '鄉鎮市區', '使用分區', '主要用途', '主要建材', '建物型態'], dummy_na=True) #'縣市', '鄉鎮市區', 

# %%
float_df = enc_df.astype(np.float64)
whole_df = float_df.copy()


# %%
# standardize input_df only
input_df = whole_df.copy()
label_df = input_df.pop('單價').to_frame()

n_features = len(input_df.columns)
input_means = input_df.mean()
input_stds = input_df.std()
input_df = (input_df - input_means) / (input_stds + 1e-15)


# %%
whole_idxs = np.arange(len(whole_df))
train_idxs, valid_idxs = train_test_split(whole_idxs, train_size=0.9)
train_inputs = tc.tensor(input_df.to_numpy()[train_idxs])
train_labels = tc.tensor(label_df.to_numpy()[train_idxs])
valid_inputs = tc.tensor(input_df.to_numpy()[valid_idxs])
valid_labels = tc.tensor(label_df.to_numpy()[valid_idxs])

# %%
hidden_size = (64, 32, 16)
model = DNN(input_size=n_features, output_size=1, hidden_size=hidden_size)


# %%
lr = 3e-5
n_epochs = 100
batch_size = 1024
loss_name = 'MAPE'
metric_names = ['MAPE']
optimizer = Adam(model.parameters(), lr=lr)
log_dict = train_model(
    train_inputs, 
    train_labels, 
    valid_inputs, 
    valid_labels, 
    model, 
    optimizer, 
    n_epochs, 
    batch_size,
    loss_name,
    metric_names
)

epoch_idxs = np.arange(1, n_epochs + 1)
for log_name, log_values in log_dict.items():
    train_log_values = log_values['train']
    valid_log_values = log_values['valid']
    plt.figure(figsize=(10, 10))
    plt.title(f'{log_name}', fontsize=20)
    plt.plot(epoch_idxs, train_log_values, label='Train', c='c')
    plt.plot(epoch_idxs, valid_log_values, label='Valid', c='b')
    plt.xlabel('Epoch', fontsize=20)
    plt.legend()
    plt.show()

# %%
epoch_idxs = np.arange(1, n_epochs + 1)
for log_name, log_values in log_dict.items():
    train_log_values = log_values['train']
    valid_log_values = log_values['valid']
    plt.figure(figsize=(10, 10))
    plt.title(f'{log_name}', fontsize=20)
    plt.plot(epoch_idxs, train_log_values, label='Train', c='c')
    plt.plot(epoch_idxs, valid_log_values, label='Valid', c='b')
    plt.xlabel('Epoch', fontsize=20)
    plt.legend()
    plt.ylim(0.1, 0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
# %%
