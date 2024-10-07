# %% 
# import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from metric import np_MAPE, np_RMSE


# %%
orig_df = pd.read_csv('30_Training Dataset_V2/training_data.csv', header=0, keep_default_na=False, na_values=['None', '', '其他'])


# %%
enc_df = pd.get_dummies(orig_df, columns=['使用分區', '主要用途', '主要建材', '建物型態'], dummy_na=True)


# %%
numeric_df = enc_df.drop(columns=['ID', '縣市', '鄉鎮市區', '路名', '備註']).astype(np.float64)


# %%
# std data
means = numeric_df.mean()
stds = numeric_df.std()
input_df = (numeric_df - means) / (stds + 1e-15)
label_df = input_df.pop('單價').to_frame()
label_mean = means['單價']
label_std = stds['單價']


# %%
def unstd_data(data, mean, std):
    return mean + data * std

# %%
# fit whole df
train_inputs = input_df.to_numpy()
train_labels = label_df.to_numpy()
reg_fn = LinearRegression().fit(train_inputs, train_labels)
train_preds = reg_fn.predict(train_inputs)

unstd_train_labels = unstd_data(train_labels, label_mean, label_std)
unstd_train_preds = unstd_data(train_preds, label_mean, label_std)
fit_all_MAPE = np_MAPE(unstd_train_labels, unstd_train_preds)
fit_all_RMSE = np_RMSE(unstd_train_labels, unstd_train_preds)
print(fit_all_MAPE)
print(fit_all_RMSE)


# %%
# fit each fold
n_splits = 10
train_MAPEs = np.zeros(n_splits+1)
train_RMSEs = np.zeros(n_splits+1)
train_MAPEs[0] = fit_all_MAPE
train_RMSEs[0] = fit_all_RMSE
test_MAPEs = np.zeros(n_splits+1)
test_RMSEs = np.zeros(n_splits+1)
kf = KFold(n_splits=n_splits, shuffle=True)
splits = kf.split(X=input_df, y=label_df)
for i, (train_idxs, test_idxs) in enumerate(splits):
    train_inputs = input_df.to_numpy()[train_idxs]
    train_labels = label_df.to_numpy()[train_idxs]
    test_inputs = input_df.to_numpy()[test_idxs]
    test_labels = label_df.to_numpy()[test_idxs]

    reg_fn = LinearRegression().fit(train_inputs, train_labels)

    train_preds = reg_fn.predict(train_inputs)
    unstd_train_labels = unstd_data(train_labels, label_mean, label_std)
    unstd_train_preds = unstd_data(train_preds, label_mean, label_std)
    train_MAPE = np_MAPE(unstd_train_labels, unstd_train_preds)
    train_RMSE = np_RMSE(unstd_train_labels, unstd_train_preds)
    train_MAPEs[i + 1] = train_MAPE
    train_RMSEs[i + 1] = train_RMSE

    test_preds = reg_fn.predict(test_inputs)
    unstd_test_labels = unstd_data(test_labels, label_mean, label_std)
    unstd_test_preds = unstd_data(test_preds, label_mean, label_std)
    test_MAPE = np_MAPE(unstd_test_labels, unstd_test_preds)
    test_RMSE = np_RMSE(unstd_test_labels, unstd_test_preds)
    test_MAPEs[i + 1] = test_MAPE
    test_RMSEs[i + 1] = test_RMSE


split_idxs = np.arange(0, n_splits + 1)
metric_df = pd.DataFrame({
    'split_idx': np.tile(split_idxs, 2),
    'type': ['train' for _ in range(n_splits + 1)] + ['test' for _ in range(n_splits + 1)],
    'MAPE': np.concatenate((train_MAPEs, test_MAPEs))*100,
    'RMSE': np.concatenate((train_RMSEs, test_RMSEs))
})

plt.title('LR on encode data (std all and metric unstd labels)')
sns.barplot(metric_df, x='split_idx', y='MAPE', hue='type')
plt.ylabel('MAPE (%)')
plt.ylim(0, 50)
plt.show()

plt.title('LR on encode data (std all and metric unstd labels)')
sns.barplot(metric_df, x='split_idx', y='RMSE', hue='type')
plt.ylim(0, 1.5)
plt.show()

# %%
