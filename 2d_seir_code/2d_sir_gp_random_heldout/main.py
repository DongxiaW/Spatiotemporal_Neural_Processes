# -*- coding: utf-8 -*-
"""2D_iclr2022_SIR_GP_random_heldout.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xGBWjV6R1FQ6p7SPlRaJcymk1ZG1laXK
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from numpy.random import binomial
import torch
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import torch.nn as nn
from sklearn import preprocessing
from scipy.stats import multivariate_normal
import math
import gpytorch

device = torch.device("cpu")

large = 25; med = 19; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': 20,
          'figure.figsize': (27, 8),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': med}
plt.rcParams.update(params)

"""# generate y ( sequence) using SEIR model:"""

def seir (num_days, beta_epsilon_flatten, num_simulations):
    mu = 1 #0.4
    all_cmpts = ['S', 'E', 'I', 'R', 'F']
    all_cases = ['E', 'I', 'R', 'F']

    x = range(num_days)

    ## model parameters
    ## initialization of california
    N = 100000 #39512000

    ## save the number of death (mean, std) for each senario
    train_mean_list = []
    train_std_list = []
    train_list = []

    for i in range (len(beta_epsilon_flatten)):
        init_I = int(2000) #2000
        init_R = int(0) #0
        init_E = int(2000) #2000

        ## save the number of individuals in each cmpt everyday
        dic_cmpts = dict()
        for cmpt in all_cmpts:
            dic_cmpts[cmpt] = np.zeros((num_simulations, num_days)).astype(int)

        dic_cmpts['S'][:, 0] = N - init_I - init_R - init_E
        dic_cmpts['I'][:, 0] = init_I
        dic_cmpts['E'][:, 0] = init_E
        dic_cmpts['R'][:, 0] = init_R
        

        ## save the number of new individuals entering each cmpt everyday
        dic_cases = dict()
        for cmpt in all_cmpts[1:]:
            dic_cases[cmpt] = np.zeros((num_simulations, num_days))

        ## run simulations
        for simu_id in range(num_simulations):
            for t in range(num_days-1):
                ## SEIR: stochastic
                flow_S2E = binomial(dic_cmpts['S'][simu_id, t], beta_epsilon_flatten[i,0] * dic_cmpts['I'][simu_id, t] / N)
                flow_E2I = binomial(dic_cmpts['E'][simu_id, t], beta_epsilon_flatten[i,1])
                flow_I2R = binomial(dic_cmpts['I'][simu_id, t], mu)
#                 print(t,flow_R2F)
                dic_cmpts['S'][simu_id, t+1] = dic_cmpts['S'][simu_id, t] - flow_S2E
                dic_cmpts['E'][simu_id, t+1] = dic_cmpts['E'][simu_id, t] + flow_S2E - flow_E2I
                dic_cmpts['I'][simu_id, t+1] = dic_cmpts['I'][simu_id, t] + flow_E2I - flow_I2R
                dic_cmpts['R'][simu_id, t+1] = dic_cmpts['R'][simu_id, t] + flow_I2R
                # dic_cmpts['F'][simu_id, t+1] = dic_cmpts['F'][simu_id, t] + flow_R2F

            
                ## get new cases per day
                dic_cases['E'][simu_id, t+1] = flow_S2E # exposed
                dic_cases['I'][simu_id, t+1] = flow_E2I # infectious
                dic_cases['R'][simu_id, t+1] = flow_I2R # removed
                # dic_cases['F'][simu_id, t+1] = flow_R2F # death 
        
        # rescale_cares_E = dic_cmpts['E'][...,1:]/N
        rescale_cares_I = dic_cmpts['I'][...,1:]/N*100
        # rescale_cares_R = dic_cmpts['R'][...,1:]/N

        train_list.append(rescale_cares_I)
        train_mean_list.append(np.mean(rescale_cares_I,axis=0))
        train_std_list.append(np.std(rescale_cares_I,axis=0))        

    train_meanset = np.stack(train_mean_list,0)
    train_stdset = np.stack(train_std_list,0)
    train_set = np.stack(train_list,0)
    return train_set, train_meanset, train_stdset

num_days = 101
num_simulations = 30
beta = np.repeat(np.expand_dims(np.linspace(1.1, 4.0, 30),1),9,1)
epsilon = np.repeat(np.expand_dims(np.linspace(0.25, 0.65, 9),0),30,0)
beta_epsilon = np.stack([beta,epsilon],-1)
beta_epsilon_train = beta_epsilon.reshape(-1,2)

beta = np.repeat(np.expand_dims(np.linspace(1.24, 3.98, 5),1),3,1)
epsilon = np.repeat(np.expand_dims(np.linspace(0.31, 0.61, 3),0),5,0)
beta_epsilon = np.stack([beta,epsilon],-1)
beta_epsilon_test = beta_epsilon.reshape(-1,2)

"""# GP"""

# Commented out IPython magic to ensure Python compatibility.

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=100
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=0.5), num_tasks=100, rank=1
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train(training_iterations,x_train,y_train, patience = 500):

    # Find optimal model hyperparameters
    min_loss = 0. # for early stopping
    wait = 0
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=1e-2)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        if (i % 500)==0:
            print('Iter %d/%d - Loss: %.3f' % (i, training_iterations, loss.item()))
        if loss  < min_loss:
            wait = 0
            min_loss = loss

        elif loss >= min_loss:
            wait += 1
            if wait == patience:
                print('Early stopping at iteration: %d' % i)
                return 0
    return 0 

def test(x_test):

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # x_test = torch.linspace(1.1, 4., 59).to(device)
        predictions = likelihood(model(x_test))
        mean = predictions.mean
        # lower, upper = predictions.confidence_region()
        std = predictions.stddev

    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

    return mean, std

def select_data(x_train, y_train, beta_epsilon_all, yall_set, score_array, selected_mask):

    mask_score_array = score_array*(1-selected_mask)
    # print('mask_score_array',mask_score_array)
    select_index = np.argmax(mask_score_array)
    print('select_index:',select_index)


    selected_x = beta_epsilon_all[select_index:select_index+1]
    selected_y = yall_set[select_index]

    x_train1 = np.repeat(selected_x,num_simulations,axis =0)
    x_train = np.concatenate([x_train, x_train1],0)
    
    y_train1 = selected_y.reshape(-1,100)
    y_train = np.concatenate([y_train, y_train1],0)
 
    selected_mask[select_index] = 1
    
    return x_train, y_train, selected_mask

def calculate_score(x_train, y_train, beta_epsilon_all):

    score_array = np.array(range(1,len(beta_epsilon_all)+1))
    np.random.shuffle(score_array)

    return score_array

def sample(y_mean,y_std,num_simulations):
    y_mean = np.repeat(y_mean,num_simulations, 0)
    y_std = np.repeat(y_std,num_simulations, 0)
    output = np.random.normal(y_mean, y_std)
    
    return output

"""BO search:"""

def mae_plot(mae, selected_mask,i,j):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    selected_mask = selected_mask.reshape(30,9)
    mae_min, mae_max = 0, 1200

    fig, ax = plt.subplots(figsize=(16, 7))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, mae, cmap='binary', vmin=mae_min, vmax=mae_max)
    ax.set_title('MAE Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.savefig('mae_plot_seed%d_itr%d.pdf' % (i,j))
    plt.close(fig)

def score_plot(score, selected_mask,i,j):
    epsilon, beta  = np.meshgrid(np.linspace(0.25, 0.7, 10), np.linspace(1.1, 4.1, 31))
    score_min, score_max = 0, 1
    selected_mask = selected_mask.reshape(30,9)
    score = score.reshape(30,9)
    fig, ax = plt.subplots(figsize=(16, 7))
    # f, (y1_ax) = plt.subplots(1, 1, figsize=(16, 10))

    c = ax.pcolormesh(beta-0.05, epsilon-0.025, score, cmap='binary', vmin=score_min, vmax=score_max)
    ax.set_title('Score Mesh')
    # set the limits of the plot to the limits of the data
    ax.axis([beta.min()-0.05, beta.max()-0.05, epsilon.min()-0.025, epsilon.max()-0.025])
    x,y = np.where(selected_mask==1)
    x = x*0.1+1.1
    y = y*0.05+0.25
    ax.plot(x, y, 'r*', markersize=15)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Epsilon')
    plt.savefig('score_plot_seed%d_itr%d.pdf' % (i,j))
    plt.close(fig)

def MAE_MX(y_pred, y_truth):
    N = 100000
    y_pred = y_pred.reshape(30,9, 30, 100)*N/100
    y_truth = y_truth.reshape(30,9, 30, 100)*N/100
    mae_matrix = np.mean(np.abs(y_pred - y_truth),axis=(2,3))
    mae = np.mean(np.abs(y_pred - y_truth))
    return mae_matrix, mae

def MAE(y_pred, y_truth):
    N = 100000
    y_pred = y_pred*N/100
    y_truth = y_truth*N/100
    return np.mean(np.abs(y_pred-y_truth))

beta_epsilon_all = beta_epsilon_train
yall_set, yall_mean, yall_std = seir(num_days,beta_epsilon_all,num_simulations)
y_all = yall_set.reshape(-1,100)
x_all = np.repeat(beta_epsilon_all,num_simulations,axis =0)


ytest_set, ytest_mean, ytest_std = seir(num_days,beta_epsilon_test,num_simulations)
y_test = ytest_set.reshape(-1,100)
x_test = np.repeat(beta_epsilon_test,num_simulations,axis =0)

np.random.seed(3)
mask_init = np.zeros(len(beta_epsilon_all))
mask_init[:2] = 1

np.random.shuffle(mask_init)
selected_beta_epsilon = beta_epsilon_all[mask_init.astype('bool')]
x_train_init = np.repeat(selected_beta_epsilon,num_simulations,axis =0)

selected_y = yall_set[mask_init.astype('bool')]
y_train_init = selected_y.reshape(selected_y.shape[0]*selected_y.shape[1],selected_y.shape[2])

N = 100000 #population

ypred_allset = []
ypred_testset = []
mae_allset = []
maemetrix_allset = []
mae_testset = []
score_set = []
mask_set = []

for seed in range(3): #3
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    y_pred_test_list = []
    y_pred_all_list = []
    all_mae_matrix_list = []
    all_mae_list = []
    test_mae_list = []
    score_list = []
    mask_list = []

    x_train,y_train = x_train_init, y_train_init
    selected_mask = np.copy(mask_init)

    for i in range(10): #8
        # print('selected_mask:', selected_mask)
        print('training data shape:', x_train.shape, y_train.shape)
        mask_list.append(np.copy(selected_mask))
    
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=100).to(device)
        model = MultitaskGPModel(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), likelihood).to(device)
        losses = train(30000,torch.from_numpy(x_train).float().to(device),torch.from_numpy(y_train).float().to(device),500) 

        model.eval()
        likelihood.eval()

        y_pred_test_mean,y_pred_test_std = test(torch.from_numpy(x_test.reshape(-1,30,2)[:,1]).float())
        y_pred_test = sample(y_pred_test_mean,y_pred_test_std,num_simulations)
        y_pred_test_list.append(y_pred_test)

        test_mae = MAE(y_pred_test,y_test)
        test_mae_list.append(test_mae)
        print('Test MAE:',test_mae)

        y_pred_all_mean,y_pred_all_std = test(torch.from_numpy(x_all.reshape(-1,30,2)[:,1]).float())
        y_pred_all = sample(y_pred_all_mean,y_pred_all_std,num_simulations)
        y_pred_all_list.append(y_pred_all)
        mae_matrix, mae = MAE_MX(y_pred_all, y_all)
        
        
        all_mae_matrix_list.append(mae_matrix)
        all_mae_list.append(mae)
        print('All MAE:',mae)
        mae_plot(mae_matrix, selected_mask,seed,i)

        score_array = calculate_score(x_train, y_train, beta_epsilon_all)
        score_array = (score_array - np.min(score_array))/(np.max(score_array) - np.min(score_array))
        
        score_list.append(score_array)
        score_plot(score_array, selected_mask,seed,i)

        x_train, y_train, selected_mask = select_data(x_train, y_train, beta_epsilon_all, yall_set, score_array, selected_mask)

    y_pred_all_arr = np.stack(y_pred_all_list,0)
    y_pred_test_arr = np.stack(y_pred_test_list,0)
    all_mae_matrix_arr = np.stack(all_mae_matrix_list,0)
    all_mae_arr = np.stack(all_mae_list,0)
    test_mae_arr = np.stack(test_mae_list,0)
    score_arr = np.stack(score_list,0)
    mask_arr = np.stack(mask_list,0)

    ypred_allset.append(y_pred_all_arr)
    ypred_testset.append(y_pred_test_arr)
    maemetrix_allset.append(all_mae_matrix_arr)
    mae_allset.append(all_mae_arr)
    mae_testset.append(test_mae_arr)
    score_set.append(score_arr)
    mask_set.append(mask_arr)

ypred_allarr = np.stack(ypred_allset,0)
ypred_testarr = np.stack(ypred_testset,0) 
maemetrix_allarr = np.stack(maemetrix_allset,0) 
mae_allarr = np.stack(mae_allset,0)
mae_testarr = np.stack(mae_testset,0)
score_arr = np.stack(score_set,0)
mask_arr = np.stack(mask_set,0)

np.save('mae_testarr.npy',mae_testarr)
np.save('mae_allarr.npy',mae_allarr)
np.save('maemetrix_allarr.npy',maemetrix_allarr)

np.save('score_arr.npy',score_arr)
np.save('mask_arr.npy',mask_arr)

np.save('y_pred_all_arr.npy',ypred_allarr)
np.save('y_pred_test_arr.npy',ypred_testarr)

np.save('y_all.npy',y_all)
np.save('y_test.npy',y_test)

