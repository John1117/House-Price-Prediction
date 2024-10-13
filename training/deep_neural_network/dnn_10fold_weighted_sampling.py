# %%
# Import
import numpy as np 
import pandas as pd
import torch as tc
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from metric import tc_MAPE, tc_RMSE, tc_MAE, tc_MSE, tc_RRSE, tc_RSE



# %%
metric_dict = {'MAPE': tc_MAPE, 'RMSE': tc_RMSE, 'MAE': tc_MAE, 'MSE': tc_MSE, 'RRSE': tc_RRSE, 'RSE': tc_RSE}




# %%
orig_df = pd.read_csv('30_Training Dataset_V2/training_data.csv', header=0, keep_default_na=False, na_values=['None', '', '其他'])





# %%
enc_df = pd.get_dummies(orig_df, columns=['使用分區', '主要用途', '主要建材', '建物型態'], dummy_na=True)
df = enc_df.drop(columns=['ID', '縣市', '鄉鎮市區', '路名', '備註']).astype(np.float64)



# %%
unit_price_counts_df = df.value_counts(subset='單價', sort=True)
# unit_price_counts_df.index


# %%
unit_price_counts_series = df.apply(func=lambda row: unit_price_counts_df[row['單價']], axis=1)
all_sample_weights = 1 / unit_price_counts_series.values




# %%
means = df.mean()
stds = df.std()
input_df = (df - means) / (stds + 1e-15)
label_df = input_df.pop('單價').to_frame()
label_mean = means['單價']
label_std = stds['單價']


def unstd_data(data, mean, std):
    return mean + data * std



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
            if i==len(hidden_size)-1:
                self.seq.append(nn.Linear(hidden_size[i], output_size, dtype=dtype))
            else:
                self.seq.append(nn.Linear(hidden_size[i], hidden_size[i+1], dtype=dtype))
                self.seq.append(nn.ReLU())

    def forward(self, input):
        return self.seq(input)
    



# %%
def train_model(train_inputs, train_labels, test_inputs, test_labels, model, lr=1e-3, n_epochs=10, batch_size=64, loss_fn=tc_MSE, sample_weights=None):
    optimr = Adam(model.parameters(), lr=lr)
    shuffle = True
    sampler = None
    if sample_weights is not None:
        shuffle = False
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=batch_size)

    data_loader = DataLoader(
        TensorDataset(train_inputs, train_labels), 
        batch_size=batch_size,
        shuffle=shuffle, 
        sampler=sampler
        )

    train_MAPEs = np.zeros(n_epochs)
    train_losses = np.zeros(n_epochs)
    test_MAPEs = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    for i in range(n_epochs):
        model.train()
        for data in data_loader:
            inputs, labels = data
            optimr.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimr.step()

        model.eval()
        with tc.no_grad():
            train_preds = model(train_inputs)
            unstd_train_labels = unstd_data(train_labels, label_mean, label_std)
            unstd_train_preds = unstd_data(train_preds, label_mean, label_std)
            train_MAPE = tc_MAPE(unstd_train_labels, unstd_train_preds)
            train_loss = loss_fn(unstd_train_labels, unstd_train_preds)
            train_MAPEs[i] = train_MAPE
            train_losses[i] = train_loss

            
            test_preds = model(test_inputs)
            unstd_test_labels = unstd_data(test_labels, label_mean, label_std)
            unstd_test_preds = unstd_data(test_preds, label_mean, label_std)
            test_MAPE = tc_MAPE(unstd_test_labels, unstd_test_preds)
            test_loss = loss_fn(unstd_test_labels, unstd_test_preds)
            test_MAPEs[i] = test_MAPE
            test_losses[i] = test_loss
            print(f'Epoch {i+1}: MAPE: (train, test) = ({train_MAPE:.7f}, {test_MAPE:.7f}) loss: (train, test) = ({train_loss:.7f}, {test_loss:.7f})')

    return train_MAPEs, test_MAPEs, train_losses, test_losses



# %%
n_splits = 10
n_epochs = 1000
batch_size = 1024
lr = 1e-4
loss_fn_name = 'MSE'

epoch_idxs = np.arange(1, n_epochs + 1)
train_MAPEs = np.zeros((n_splits, n_epochs))
test_MAPEs = np.zeros((n_splits, n_epochs))
train_losses = np.zeros((n_splits, n_epochs))
test_losses = np.zeros((n_splits, n_epochs))

kf = KFold(n_splits=n_splits, shuffle=True)
splits = kf.split(X=input_df, y=label_df)
for i, (train_idxs, test_idxs) in enumerate(splits):
    print('\n')
    print(f'{i+1}-fold')
    sample_weights = all_sample_weights[train_idxs]
    train_inputs = tc.tensor(input_df.to_numpy()[train_idxs])
    train_labels = tc.tensor(label_df.to_numpy()[train_idxs])
    test_inputs = tc.tensor(input_df.to_numpy()[test_idxs])
    test_labels = tc.tensor(label_df.to_numpy()[test_idxs])
    
    dnn = DNN(input_size=len(input_df.columns), output_size=len(label_df.columns), hidden_size=64)

    dnn.eval()
    with tc.no_grad():
        train_preds = dnn(train_inputs)
        unstd_train_labels = unstd_data(train_labels, label_mean, label_std)
        unstd_train_preds = unstd_data(train_preds, label_mean, label_std)
        train_MAPE = tc_MAPE(unstd_train_labels, unstd_train_preds)
        train_RMSE = tc_RMSE(unstd_train_labels, unstd_train_preds)
        
        test_preds = dnn(test_inputs)
        unstd_test_labels = unstd_data(test_labels, label_mean, label_std)
        unstd_test_preds = unstd_data(test_preds, label_mean, label_std)
        test_MAPE = tc_MAPE(unstd_test_labels, unstd_test_preds)
        test_RMSE = tc_RMSE(unstd_test_labels, unstd_test_preds)
    print(f'Epoch 0: MAPE: (train, test) = ({train_MAPE:.7f}, {test_MAPE:.7f}) loss: (train, test) = ({train_RMSE:.7f}, {test_RMSE:.7f})')

    
    loss_fn = metric_dict[loss_fn_name]
    split_train_MAPEs, split_test_MAPEs, split_train_losses, split_test_losses = train_model(train_inputs, train_labels, test_inputs, test_labels, dnn, lr=lr, n_epochs=n_epochs, batch_size=batch_size, loss_fn=loss_fn, sample_weights=sample_weights)
    
    train_MAPEs[i] = split_train_MAPEs
    test_MAPEs[i] = split_test_MAPEs
    train_losses[i] = split_train_losses
    test_losses[i] = split_test_losses


    plt.title(f'Epoch-{i+1} MAPE', fontsize=20)
    plt.plot(epoch_idxs, split_train_MAPEs, label='Train', c='c')
    plt.plot(epoch_idxs, split_test_MAPEs, label='Test', c='b')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylim(0.25, 0.5)
    plt.legend()
    plt.show()

    plt.title(f'Epoch-{i+1} {loss_fn_name}Loss', fontsize=20)
    plt.plot(epoch_idxs, split_train_losses, label='Train', c='c')
    plt.plot(epoch_idxs, split_test_losses, label='Test', c='b')
    plt.xlabel('Epoch', fontsize=20)
    plt.legend()
    plt.show()


train_MAPE_means = train_MAPEs.mean(axis=0)
train_MAPE_stds = train_MAPEs.std(axis=0)
test_MAPE_means = test_MAPEs.mean(axis=0)
test_MAPE_stds = test_MAPEs.std(axis=0)

train_loss_means = train_losses.mean(axis=0)
train_loss_stds = train_losses.std(axis=0)
test_loss_means = test_losses.mean(axis=0)
test_loss_stds = test_losses.std(axis=0)

plt.title('Avg MAPE', fontsize=20)
plt.plot(epoch_idxs, train_MAPE_means, color=(0, 1, 1), label='Train')
plt.plot(epoch_idxs, test_MAPE_means, color=(0, 0, 1), label='Test')
plt.fill_between(epoch_idxs, train_MAPE_means - train_MAPE_stds, train_MAPE_means + train_MAPE_stds, color=(0, 1, 1, 0.2))
plt.fill_between(epoch_idxs, test_MAPE_means - test_MAPE_stds, test_MAPE_means + test_MAPE_stds, color=(0, 0, 1, 0.2))
plt.xlabel('Epoch', fontsize=20)
plt.ylim(0.25, 0.5)
plt.legend()
plt.show()
plt.savefig(f'10-fold_metric_test/{loss_fn_name}Loss_MAPE')

plt.title(f'Avg {loss_fn_name}Loss', fontsize=20)
plt.plot(epoch_idxs, train_loss_means, color=(0, 1, 1), label='Train')
plt.plot(epoch_idxs, test_loss_means, color=(0, 0, 1), label='Test')
plt.fill_between(epoch_idxs, train_loss_means - train_loss_stds, train_loss_means + train_loss_stds, color=(0, 1, 1, 0.2))
plt.fill_between(epoch_idxs, test_loss_means - test_loss_stds, test_loss_means + test_loss_stds, color=(0, 0, 1, 0.2))
plt.xlabel('Epoch', fontsize=20)
plt.legend()
plt.show()
plt.savefig(f'10-fold_metric_test/{loss_fn_name}Loss_{loss_fn_name}Loss')



# %%
