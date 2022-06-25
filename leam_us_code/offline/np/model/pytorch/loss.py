import torch
import torch.nn as nn 
from scipy.stats import multivariate_normal
import numpy as np


def mae_loss(y_pred, y_true):
    loss = torch.abs(y_pred - y_true)
    loss[loss != loss] = 0
    return loss.mean()

def mae_metric(y_pred, y_true):
    loss = np.abs(y_pred - y_true)
    loss[loss != loss] = 0
    loss = np.mean(loss)
    return loss

def rmse_metric(y_pred, y_true):
    loss0 = (y_pred-y_true)**2
    loss0[loss0 != loss0] = 0
    loss = np.sqrt(np.mean(loss0))
    return loss
