# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:22:50 2017

@author: jtmccoy

Code to test mean replacement and PCA imputation for missing data imputation.

Anglo Platinum Literature Review 2
JT McCoy, L Auret December 2017
Department of Process Engineering, Stellenbosch University

"""

import itertools
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

resdpi = 150
random.seed(a=1) # specify seed for reproducibility

# Load data from a csv for analysis:
Xdata_df = pd.read_csv("milldata.csv")
Xdata = Xdata_df.values
del Xdata_df

# Load data with missing values from a csv for analysis:
Xdata_df = pd.read_csv("milldatacorruptheavy.csv")
Xdata_Missing = Xdata_df.values
del Xdata_df

# Find indices of missing and observed values in Xdata_Missing:
NanIndex = np.where(np.isnan(Xdata_Missing))
ObsIndex = np.where(np.isfinite(Xdata_Missing))
ObsRowInd = np.where(np.isfinite(np.sum(Xdata_Missing,axis=1)))
NanRowInd = np.where(np.isnan(np.sum(Xdata_Missing,axis=1)))
Xdata_Missing_Rows = NanRowInd[0] # number of rows with missing values
# Number of missing values
NanCount = len(NanIndex[0])

# Replace missing values with random values:
# Determine min/max of observed data:
Xdata_min = min(Xdata_Missing[ObsIndex])
Xdata_max = max(Xdata_Missing[ObsIndex])
# generate a sequence of values within the min/max from which to sample:
SeqCount = NanCount*100
Xdata_random_vector = Xdata_min + (Xdata_max - Xdata_min)*range(SeqCount)/SeqCount
# sample with replacement from the sequence and replace NaNs with these values:
Xdata_Missing[NanIndex] = random.choices(Xdata_random_vector,k=NanCount)

Xdata_Missing[NanIndex] = 0
NumVars = Xdata_Missing.shape[1]

# begin PCA imputation:
ReconError = 1000
IterCount = 0
MaxIter = 100
ReconErrVec = np.zeros(MaxIter)
RetComp = np.zeros(MaxIter)
MissVal = np.zeros([MaxIter,NanCount], dtype=np.float32)
RetainVar = 0.3 # Variance to retain in model

while IterCount < MaxIter:
    # Feature Scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(Xdata_Missing)
    
    # fit all PCs:
    pca = PCA(n_components = NumVars)
    pca.fit(X_scaled)
    
    # Determine number of PCs to retain, keeping RetainVar variance:
    explained_variance = pca.explained_variance_ratio_
    explained_variance = np.cumsum(explained_variance)
    RetainedComp = np.argmax(explained_variance>=RetainVar)+1
    RetComp[IterCount] = RetainedComp
#    RetainedComp = 7
    
    # fit PCA with RetainedComp components:
    pca = PCA(n_components = RetainedComp)
    X_PCA = pca.fit_transform(X_scaled)
    X_hat = pca.inverse_transform(X_PCA)
    
    # Reconstruction error:
    ReconError = LA.norm(X_scaled[ObsIndex] - X_hat[ObsIndex])
    
    # Replace missing values with reconstructed values:
    Xdata_Missing[NanIndex] = sc.inverse_transform(X_hat)[NanIndex]
    
    ReconErrVec[IterCount] = ReconError
    MissVal[IterCount,:] = Xdata_Missing[NanIndex]
    IterCount = IterCount + 1

fig = plt.figure(dpi = resdpi)
plt.plot(ReconErrVec)
plt.xlabel('Iteration')
plt.ylabel('Observed value reconstruction error')
plt.show()
#fig.savefig('milldata PCA Recon', bbox_inches = 'tight')

#fig = plt.figure(dpi = resdpi)
#if NanCount == 1:
#    plt.plot(range(MaxIter),MissVal[:,0],'-.')
#    plt.plot([0, MaxIter],[Xdata[NanIndex], Xdata[NanIndex]])
#    plt.xlabel('Iteration')
#    plt.ylabel('Missing value')
#else:
#    subplotmax = min(NanCount,3)
#    for plotnum in range(subplotmax):
#        TrueVal = Xdata[NanIndex[0][plotnum]][NanIndex[1][plotnum]]
#        plt.subplot(subplotmax,1,plotnum+1)
#        plt.plot(range(MaxIter),MissVal[:,plotnum],'-.')
#        plt.plot([0, MaxIter],[TrueVal, TrueVal])
#        plt.xlabel('Iteration')
#        plt.ylabel('Missing value ' + str(plotnum+1))
#plt.show()
#fig.savefig('milldata PCA Impute', bbox_inches = 'tight')

# Comparison of reconstruction error:

# Properties of data:
Xdata_length = Xdata_Missing.shape[0] # number of data points to use
n_x = Xdata_Missing.shape[1] # dimensionality of data space

## Comparison of reconstruction error:
## Zscore of the original complete records used to standardise imputed data:
#sc_error = StandardScaler()
#Xdata_Missing_complete = np.copy(Xdata_Missing[ObsRowInd[0],:])
## standardise using complete records:
#sc_error.fit(Xdata_Missing_complete)
#Xdata_Missing = sc_error.transform(Xdata_Missing)
#del Xdata_Missing_complete

# plot imputation results for one variable:
var_i = 0
min_i = np.min(Xdata[:,var_i])
max_i = np.max(Xdata[:,var_i])

fig = plt.figure(dpi = 150)
plt.plot(Xdata[NanIndex[0][np.where(NanIndex[1]==var_i)],var_i],Xdata_Missing[NanIndex[0][np.where(NanIndex[1]==var_i)],var_i],'.')
plt.plot([min_i, max_i], [min_i, max_i])
plt.xlabel('True value')
plt.ylabel('Imputed value')
plt.show()

# Standardise wrt original Xdata for RMSE calculation
sc_error = StandardScaler()
sc_error.fit(Xdata)
Xdata_Missing = sc_error.transform(Xdata_Missing)
Xdata = sc_error.transform(Xdata)

ReconstructionError_baseline = sum(((Xdata[NanIndex])**2)**0.5)/NanCount
print('Reconstruction error (replace with mean):')
print(ReconstructionError_baseline)

ReconstructionError = sum(((Xdata_Missing[NanIndex] - Xdata[NanIndex])**2)**0.5)/NanCount
print('Reconstruction error (iterated PCA imputation):')
print(ReconstructionError)

print(RetComp.mean())