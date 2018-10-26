"""
Main file to run VAE for missing data imputation, Presented at IFAC MMM2018
by JT McCoy, RS Kroon and L Auret.

This file learns a one step ahead prediction of the data in a csv, allowing
the prediction of the most likely next values given the current observation.

Based on implementations
of VAEs from:
    https://github.com/twolffpiggott/autoencoders
    https://jmetzen.github.io/2015-11-27/vae.html
    https://github.com/lazyprogrammer/machine_learning_examples/blob/master/unsupervised_class3/vae_tf.py
    https://github.com/deep-learning-indaba/practicals2017/blob/master/practical5.ipynb

VAE is designed to handle real-valued data, not binary data, so the source code
has been adapted to work only with Gaussians as the output of the generative
model (p(x|z)).

"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencoders import TFVariationalAutoencoder
import pandas as pd
import random

'''
==============================================================================
'''
# DEFINE HYPERPARAMETERS
# Path to data:
DataPath = "data.csv"

# VAE network size:
Decoder_hidden1 = 20
Decoder_hidden2 = 10
Encoder_hidden1 = 10
Encoder_hidden2 = 20

# dimensionality of latent space:
latent_size = 4

# number of lagged variables:
n_lag = 2 # n_lag = 2 corresponds to one step ahead prediction
          # increase this if you want to include more observations 
          # in one step ahead prediction

# training parameters:
training_epochs = 250
batch_size = 500
learning_rate = 0.1

# specify number of imputation iterations:
ImputeIter = 25
'''
==============================================================================
'''
# LOAD DATA
# Load data from a csv for analysis:
Xdata_df = pd.read_csv(DataPath)
Xdata = Xdata_df.values
del Xdata_df

# Properties of data:
N = Xdata.shape[0] # number of data points to use
n_x = Xdata.shape[1] # dimensionality of data space

# Separate training and testing data:
test_frac = 0.1
test_ind = N - int(np.floor(N*test_frac)) # starting index of test data

data_train = Xdata.copy()
data_train = data_train[0:test_ind,:]

data_test = Xdata.copy()
data_test = data_test[test_ind:,:]

# Zscore of data wrt training data:
sc = StandardScaler()
data_train = sc.fit_transform(data_train)
data_test = sc.transform(data_test)

def next_batch(Xdata, batch_size, lag_size, MissingVals = False):
    """ Sample lagged sets of observations from data in Xdata.
        Xdata is an [NxM] matrix, N observations of M variables.
        
        Returns Xdata_sample, a [batch_size x (lag_size x M)] matrix.
    """
    if MissingVals:
        # This returns records with any missing values replaced by 0:
        Xdata_length = Xdata.shape[0]
        M = Xdata.shape[1]
        X_indices = random.sample(range(lag_size-1,Xdata_length),batch_size)
        
        Xdata_sample = np.zeros((batch_size,lag_size*M))
        for i in range(batch_size):
            Xi = X_indices[i]
            lag_i = np.arange(Xi-lag_size+1, Xi+1)
            lag_var = np.copy(Xdata[lag_i,:]).flatten()
            
            Xdata_sample[i,:] = lag_var
        
        NanIndex = np.where(np.isnan(Xdata_sample))
        Xdata_sample[NanIndex] = 0
    else:
        # This returns complete records only:
        ObsIndex = np.where(np.isfinite(np.sum(Xdata[lag_size-1:,:],axis=1)))
        M = Xdata.shape[1]
        
        # sample from the observed indices:
        X_indices = random.sample(list(ObsIndex[0]),batch_size)
        
        Xdata_sample = np.zeros((batch_size,lag_size*M))
        for i in range(batch_size):
            Xi = X_indices[i]
            lag_i = np.arange(Xi-lag_size+1, Xi+1)
            lag_var = np.copy(Xdata[lag_i,:]).flatten()
            
            Xdata_sample[i,:] = lag_var
    
    return Xdata_sample


'''
==============================================================================
'''
# INITIALISE AND TRAIN VAE
# define dict for network structure:
network_architecture = \
    dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
         n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
         n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
         n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
         n_input=n_x*n_lag, # data input size, lags of all variables except PSE
         n_z=latent_size)  # dimensionality of latent space

# initialise VAE:
vae = TFVariationalAutoencoder(network_architecture, 
                             learning_rate=learning_rate, 
                             batch_size=batch_size,
                             lag_size = n_lag)

# train VAE on corrupted data:
vae = vae.train(XData=data_train,
                training_epochs=training_epochs)

# plot training history:
fig = plt.figure(dpi = 150)
plt.plot(vae.losshistory_epoch,vae.losshistory)
plt.xlabel('Epoch')
plt.ylabel('Evidence Lower Bound (ELBO)')
plt.show()
'''
#==============================================================================
#'''

x2 = vae.predict(data_test[0:1,:], max_iter = ImputeIter)

print(x2)
print(data_test[1,:])
#'''
#==============================================================================
#'''
## GENERATE VALUES AND PLOT HISTOGRAMS
#np_x = next_batch(data_train, batch_size = 2000, lag_size = n_lag)
## reconstruct data by sampling from distribution of reconstructed variables:
#x_hat = vae.reconstruct(np_x, sample = 'mean')
#
#plt.plot(np_x[:,0], np_x[:,7], 'x')
#plt.plot(x_hat[:,0], x_hat[:,7], '.', alpha = 0.2)
#plt.show()
#
#x_hat_prior = vae.generate(n_samples = 1000)
## eval the tensor, as this allows plotting histograms:
#x_hat_prior = x_hat_prior.eval()
#
#subplotmax = min(n_x,5)
#f, axarr = plt.subplots(subplotmax, subplotmax, sharex='col', dpi = 150)
#f.suptitle('Posterior sample')
#f.subplots_adjust(wspace = 0.3)
#for k in range(subplotmax):
#    for j in range(subplotmax):
#        if k == j:
#            axarr[k, j].hist(np_x[:,k],bins = 30, density=True)
#            axarr[k, j].hist(x_hat[:,k],bins = 30, alpha = 0.7, density=True)
#            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
#            if j == 0:
#                axarr[k, j].set_ylabel('Variable 1', fontsize=6)
#            elif j == subplotmax-1:
#                axarr[k, j].set_xlabel('Variable ' + str(subplotmax), fontsize=6)
#        else:
#            axarr[k, j].plot(np_x[:,k], np_x[:,j], '+',label = 'Data')
#            axarr[k, j].plot(x_hat[:,k], x_hat[:,j], '.', alpha = 0.2, label='Posterior')
#            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
#            if j == 0:
#                axarr[k, j].set_ylabel('Variable ' + str(k+1), fontsize=6)
#            if k == subplotmax-1:
#                axarr[k, j].set_xlabel('Variable ' + str(j+1), fontsize=6)
#
#f, axarr = plt.subplots(subplotmax, subplotmax, sharex='col', dpi = 150)
#f.suptitle('Prior sample')
#f.subplots_adjust(wspace = 0.3)
#for k in range(subplotmax):
#    for j in range(subplotmax):
#        if k == j:
#            axarr[k, j].hist(np_x[:,k],bins = 30, density=True)
#            axarr[k, j].hist(x_hat_prior[:,k],bins = 30, alpha = 0.7, density=True)
#            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
#            if j == 0:
#                axarr[k, j].set_ylabel('Variable 1', fontsize=6)
#            elif j == subplotmax-1:
#                axarr[k, j].set_xlabel('Variable ' + str(subplotmax), fontsize=6)
#        else:
#            axarr[k, j].plot(np_x[:,k], np_x[:,j], '+',label = 'Data')
#            axarr[k, j].plot(x_hat_prior[:,k], x_hat_prior[:,j], '.', alpha = 0.2, label='Prior')
#            axarr[k, j].tick_params(labelsize='xx-small', pad = 0)
#            if j == 0:
#                axarr[k, j].set_ylabel('Variable ' + str(k+1), fontsize=6)
#            if k == subplotmax-1:
#                axarr[k, j].set_xlabel('Variable ' + str(j+1), fontsize=6)
#
#vae.sess.close()