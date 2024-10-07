import numpy as np
import torch as tc
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


# torch
def MSE(preds, targets):
    return tc.mean((preds - targets) ** 2)

def RMSE(preds, targets):
    return MSE(preds, targets) ** 0.5

def RSE(preds, targets):
    return tc.sum((preds - targets) ** 2) / tc.sum((targets - tc.mean(targets)) ** 2)

def RRSE(preds, targets):
    return RSE(preds, targets) ** 0.5

def MSLE(preds, targets):
    return tc.mean((tc.log(1 + targets) - tc.log(1 + preds)) ** 2)

def RMSLE(preds, targets):
    return MSLE(preds, targets) ** 0.5

def MAE(preds, targets):
    return tc.mean(tc.abs(preds - targets))

def MAPE(preds, targets):
    eps = tc.tensor(tc.finfo(targets.dtype).eps)
    return tc.mean(tc.abs(targets - preds) / tc.maximum(tc.abs(targets), eps))



# numpy
def np_MAPE(preds, targets):
    eps = np.finfo(targets.dtype).eps
    return np.mean(np.abs(targets - preds) / np.maximum(np.abs(targets), eps))

def np_RMSE(preds, targets):
    return np.mean((preds - targets) ** 2) ** 0.5




# sklearn
def skl_MAPE(preds, targets):
    return mean_absolute_percentage_error(targets, preds)

def skl_RMSE(preds, targets):
    return mean_squared_error(targets, preds, squared=False)
