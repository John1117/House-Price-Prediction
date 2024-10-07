import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
now_str = now.strftime("%Y-%m-%d %H-%M")

# %%
### MAPE LOSS
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_true, y_pred):
        # 添加一个小的常数，以防分母为零
        epsilon = 1e-5

        # 计算 MAPE
        ape = torch.abs((y_true - y_pred) / torch.maximum(torch.abs(y_true), torch.tensor(epsilon)))
        #ape = torch.abs((y_true - y_pred) / (y_true + epsilon))
        mape = torch.mean(ape)

        return mape

# %%
### model 
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super(LinearRegressionModel, self).__init__()

        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_size2, output_size)
                
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.dropout1(x) 
        x = torch.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x

# %%
### def train model 

def train_model(model:nn, train_loader, val_loader, test_loader, lr=1e-3, n_epoch=10,save=False, device= 'cpu'):
    optimr = Adam(model.parameters(), lr=lr)
    loss_fn = MAPELoss()

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    train_loss_rec=[]
    val_loss_rec=[]
    test_loss_rec=[]

    # training
    for e in range(n_epoch):
        print(f'---------第{e+1}輪訓練開始-----------')
        model.train()
        total_train_loss = 0

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            outputs = outputs.to(device)

            loss = loss_fn(outputs, labels) 
            
            # apply optim 
            optimr.zero_grad()
            loss.backward()
            optimr.step()

            total_train_loss += loss.item()

        train_loss_rec.append(total_train_loss/len(train_loader))

        print(f'round{e+1} avg train mse = {total_train_loss/ len(train_loader)}')

        # validating
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                outputs = outputs.to(device)

                loss = loss_fn(outputs, labels)
                
                total_val_loss += loss.item()

        val_loss_rec.append(total_val_loss/len(val_loader))          
        print(f'round{e+1} avg val mse ={total_val_loss/len(val_loader)}')
        
        # testing 
        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                outputs = outputs.to(device)

                loss = loss_fn(outputs, labels)
                
                total_test_loss += loss.item()
                
        test_loss_rec.append(total_test_loss/len(test_loader))       
        print(f'round{e+1} avg test mse ={total_test_loss/len(test_loader)}\n')

        if save :
            torch.save(model.state_dict(),f'G:/我的雲端硬碟/from NYCU address/For python/t-brain/save/{now_str}_model_{e + 1}.pth')
        else:
            pass

    return train_loss_rec, val_loss_rec, test_loss_rec

# %%
def plot_training_result(train_loss_rec, val_loss_rec, test_loss_rec,show = True):

    plt.plot(range(len(train_loss_rec)), train_loss_rec, label='Train Loss')
    plt.plot(range(len(val_loss_rec)), val_loss_rec, label='Validation Loss')
    plt.plot(range(len(test_loss_rec)), test_loss_rec, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.yscale('log')
    plt.ylim(0, 1)
    plt.legend()

    if show:
        plt.show()
# %%
