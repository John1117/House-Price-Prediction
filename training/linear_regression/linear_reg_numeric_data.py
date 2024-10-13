# %% 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from metric import np_MAPE, np_RMSE


# %%
orig_df = pd.read_csv('30_Training Dataset_V2/training_data.csv', header=0, keep_default_na=False, na_values=['None', ''])


# %%
dtype = np.float64
numeric_col_names = ['土地面積', '移轉層次', '總樓層數', '屋齡', '建物面積', '車位面積', '車位個數', '橫坐標', '縱坐標', '主建物面積', '陽台面積', '附屬建物面積', '單價']
numeric_df = orig_df[numeric_col_names].astype(dtype)

label_df = numeric_df.pop('單價').to_frame()
input_df = numeric_df.copy()

# %%
# fit whole df
train_inputs = input_df.to_numpy()
train_labels = label_df.to_numpy()
reg_fn = LinearRegression().fit(train_inputs, train_labels)
train_preds = reg_fn.predict(train_inputs)

fit_all_MAPE = np_MAPE(train_labels, train_preds)
fit_all_RMSE = np_RMSE(train_labels, train_preds)


# %%
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
    train_MAPE = np_MAPE(train_labels, train_preds)
    train_RMSE = np_RMSE(train_labels, train_preds)
    train_MAPEs[i + 1] = train_MAPE
    train_RMSEs[i + 1] = train_RMSE

    test_preds = reg_fn.predict(test_inputs)
    test_MAPE = np_MAPE(test_preds, test_labels)
    test_RMSE = np_RMSE(test_preds, test_labels)
    test_MAPEs[i + 1] = test_MAPE
    test_RMSEs[i + 1] = test_RMSE

split_idxs = np.arange(0, n_splits + 1)
metric_df = pd.DataFrame({
    'split_idx': np.tile(split_idxs, 2),
    'type': ['train' for _ in range(n_splits + 1)] + ['test' for _ in range(n_splits + 1)],
    'MAPE': np.concatenate((train_MAPEs, test_MAPEs))*100,
    'RMSE': np.concatenate((train_RMSEs, test_RMSEs))
})

plt.title('LR on numeric data (w/o std)')
sns.barplot(metric_df, x='split_idx', y='MAPE', hue='type')
plt.ylabel('MAPE (%)')
plt.ylim(0, 50)
plt.show()

plt.title('LR on numeric data (w/o std)')
sns.barplot(metric_df, x='split_idx', y='RMSE', hue='type')
plt.ylim(0, 1.5)
plt.show()

# %%
