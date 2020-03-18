# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:19:26 2017

@author: KUMARPX21
"""

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import stats
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import xlsxwriter
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')
#Loading Data
mat_contents=sio.loadmat('Input.mat')
C=mat_contents['A']
Input=mat_contents['Data']
Input=np.matrix(Input)
Data=Input[0:24,:]
Data=np.asmatrix(Data)

## Creating pairwise features
(m,n)=Data.shape
for i in range(0,n-1):
    for j in range(i+1,n):
        A=np.multiply(Data[:,i],Data[:,j])
        Data = np.concatenate((Data,A),axis=1)

raw_data = Data;
mean_data = np.mean(Data,axis=0)
std_data = np.std(Data,axis=0)     
Data=stats.zscore(Data,axis=0,ddof=1)
Data=np.nan_to_num(Data)

#Features = np.asmatrix(C[0,:]);
#Features.transpose()

Response=C[0:24,len(C[0])-1]
raw_response = Response
mean_resp = np.mean(Response,axis=0)
std_resp = np.std(Response,axis=0)
Response=np.matrix.transpose(stats.zscore(Response,axis=0,ddof=1))

(p,q)=Data.shape
MSE_svr = MSE_pcr = MSE_plsr = MSE_lr = 0
nlambda=10
MSE_en = np.zeros((nlambda,1))
coeff_en = np.zeros((p,q))
Pred_en = np.zeros((nlambda,p))
Pred_test = np.zeros((200,p))
training = np.zeros((p-1,1))
Y_train=np.zeros((p-1,1))
X_train=np.zeros((p-1,q))
X_test=np.zeros((1,q))

for i in range(0,len(Data)-1):
    testing=i;
    Y_test=Response[int(i)]
    X_test=Data[int(i),:]
    ct = 0
    for k in range(0,len(Data)-1):
        if(k!=i):
            ct=ct+1
            training[ct,0]=int(k)
            Y_train[ct,0]=Response[int(k)]
            X_train[ct,:]=Data[int(k),:]
    Pred_test[0,i]=Y_test*std_resp+mean_resp
    
    # Linear Regression
    lr = linear_model.LinearRegression()
    mdl_lr = lr.fit(X_train,Y_train)
    Pred_test[1,i] = mdl_lr.predict(X_test)*std_resp+mean_resp
    MSE_lr = MSE_lr + abs(Y_test-Pred_test[1,i])
    
    # Support VEctor Machine Regression
    svr_lin = SVR(kernel='linear',C=1)
    mdl_svr = svr_lin.fit(X_train,Y_train)
    Pred_test[2,i]=mdl_svr.predict(X_test)*std_resp + mean_resp          
    MSE_svr = MSE_svr + abs(Pred_test[1,i] - Y_test)
    
    # Regression Tree with Random Forest
    #X_train, X_test, y_train, y_test = train_test_split(Data, Response,train_size=p-5,random_state=4)
    #y_test = y_test*std_resp + mean_resp
    max_depth = 10
    mdl_regrf = RandomForestRegressor(max_depth=max_depth,random_state=0,n_estimators=25,max_features='auto')
    mdl_regrf.fit(X_train, Y_train)
    # Predict on test data
    Pred_test[3,i] = mdl_regrf.predict(X_test)*std_resp + mean_resp
    
    # Elastic Net Regression
    Penalty = range(0,nlambda,1);
    for j in range(0,len(Penalty)):
        enet = ElasticNet(alpha=0.1,l1_ratio=Penalty[j]/100)
        mdl_en = enet.fit(X_train,Y_train)
        Pred_en[j,i] = mdl_en.predict(X_test)*std_resp+mean_resp
        coeff_en[i,:]=enet.coef_
        MSE_en[j,0] = MSE_en[j,0] + abs(mdl_en.predict(X_test)*std_resp+mean_resp - Y_test)
    
MSE_en = MSE_en/p
MSE_svr = MSE_svr/p
MSE_lr = MSE_lr/p

min_en = np.argmin(np.min(MSE_en,axis=1))
Pred_test[4,:]=Pred_en[min_en,:]
Coeff_ENet = coeff_en[min_en,:]
r2_score_lr = r2_score(Pred_test[0,:], Pred_test[1,:])
r2_score_svr = r2_score(Pred_test[0,:], Pred_test[2,:])
r2_score_regrf = r2_score(Pred_test[0,:], Pred_test[3,:])
r2_score_en = r2_score(Pred_test[0,:], Pred_test[4,:])

fig1=plt.figure()
#plt.plot(Pred_test[0,:],Pred_test[1,:],'.',c='b',label='Linear Regression')
#plt.plot(Pred_test[0,:],Pred_test[2,:],'.',c='g',label='Support Vector Regression')
plt.plot(Pred_test[0,:],Pred_test[3,:],'.',c='r',label='Regression Tree: R^2= %f' %r2_score_regrf)
plt.plot(Pred_test[0,:],Pred_test[4,:],'.',c='k',label='Elastic Net Regression: R^2= %f' %r2_score_en)
plt.ylabel('ML Predicted')
plt.xlabel('DEM Calculated')
plt.grid(False)
plt.legend()
#plt.title("ENet R^2: %f, Regression Tree R^2: %f" % (r2_score_en, r2_score_regrf))
plt.show()
