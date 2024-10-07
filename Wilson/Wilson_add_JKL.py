# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch.nn.functional import normalize
from datetime import datetime

import Wilson_model

# device 
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# dtype
dtype = torch.float32

# now
now = datetime.now()
now_str = now.strftime("%Y-%m-%d %H-%M")

# %% 
### find if nan

PATH = 'C:/Users/Wilson/Desktop/t-brain_dataset'
df = pd.read_csv('/Users/john/personal/myself/Projects/Sinopac_HousePricePrediction/HousePricePrediction/30_Training Dataset_V2/training_data.csv')

# freq encoding
enc_j = (df.groupby('主要用途').size())/len(df)
enc_k = (df.groupby('主要建材').size())/len(df)
enc_l = (df.groupby('建物型態').size())/len(df)

df['主要用途_encode'] = df['主要用途'].apply(lambda x : enc_j[x])
df['主要建材_encode'] = df['主要建材'].apply(lambda x : enc_k[x])
df['建物型態_encode'] = df['建物型態'].apply(lambda x : enc_l[x])

#df.to_csv('encode.csv', encoding='big5')

num_rows, num_columns = df.shape

numeric_df = df.select_dtypes(include='number')
numeric_df.drop(['單價'],axis=1,inplace=True)

features = numeric_df.values
labels = df['單價'].values

# %%
### preprocess 
features_tersor = torch.tensor(features, dtype=dtype)
normalized_features = normalize(features_tersor, dim=0)

labels_tensor = torch.tensor(labels, dtype=torch.float32)
# %% 
### split data
train_ratio = 0.9
test_ratio = 0.0
val_ratio = 0.1
total_size = len(df)
batch_size = 64
shuffle = True

train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

# random_split 
train_dataset, val_dataset, test_dataset = random_split(
    TensorDataset(normalized_features, labels_tensor),
    [train_size, val_size, test_size]
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
### model = LRM
input_size = features_tersor.shape[1]
hidden_size1 = 128
hidden_size2 = 64
output_size = 1
dropout_rate = 0.0
lr = 1e-4
epoch = 15
save = False

lms_model = Wilson_model.LinearRegressionModel(
        input_size=input_size,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        output_size=output_size,
        dropout_rate=dropout_rate
)

### train and test 
train_loss_rec, val_loss_rec, test_loss_rec = Wilson_model.train_model(
                                                    model=lms_model,
                                                    train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    val_loader=val_loader,
                                                    lr=lr,
                                                    n_epoch=epoch,
                                                    save=save,
                                                    device=device
    )

### plot 
Wilson_model.plot_training_result(train_loss_rec, val_loss_rec, test_loss_rec, show=True)

# %%
### pred 
public = pd.read_csv('G:/我的雲端硬碟/from NYCU address/For python/t-brain/t-brain_dataset/public_dataset.csv')

# freq encoding
enc_j = (public.groupby('主要用途').size())/len(public)
enc_k = (public.groupby('主要建材').size())/len(public)
enc_l = (public.groupby('建物型態').size())/len(public)

public['主要用途_encode'] = public['主要用途'].apply(lambda x : enc_j[x])
public['主要建材_encode'] = public['主要建材'].apply(lambda x : enc_k[x])
public['建物型態_encode'] = public['建物型態'].apply(lambda x : enc_l[x])

public_numeric_df = public.select_dtypes(include='number')
public_features = public_numeric_df.values

# preprocess 
public_features_tersor = torch.tensor(public_features, dtype=dtype)
public_normalized_features = normalize(public_features_tersor, dim=0)
public_normalized_features = public_normalized_features.to(device)

output_list = []

lms_model.eval() 
with torch.no_grad():

    for i in range(public_normalized_features.size(0)):

        input = torch.tensor(public_normalized_features[i,:])

        output = lms_model(input)
        output.to(device)

        output_list.append(output.item())

public_numeric_df['pred_price'] = output_list
#public_numeric_df.to_csv(r'G:/我的雲端硬碟/from NYCU address/For python/t-brain/save/{}.csv'.format(now_str),encoding='big5')


# %%