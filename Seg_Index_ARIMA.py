# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:21:44 2017

@author: kumarpx21
"""
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.figure import Figure
from matplotlib.axes import Subplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import xlsxwriter

plt.style.use('fivethirtyeight')
#Loading Data
mat_contents=sio.loadmat('input.mat')
Input=mat_contents['Data']
tstart=mat_contents['start_time'];
tend=mat_contents['end_time'];
A=np.ndarray.tolist(Input[:,0])
Input=np.asanyarray(Input)
future_forecast=2000
y_forecasted1 = np.zeros((future_forecast,25))
y_truth1 = np.zeros((future_forecast,25))
y_forecasted = np.zeros((future_forecast,25))
y_truth = np.zeros((future_forecast,25))
mse_onestep=np.zeros((25,1))
mse_dynamic=np.zeros((25,1))
pred_dynamic_ci=np.zeros((future_forecast,100))
pred_ci=np.zeros((future_forecast,100))
predf_ci=np.zeros((future_forecast,100))

for i in range(0,5):
    t_start=int(tstart[i]-1)
    t_end=int(tend[i]-1)   
    y=Input[t_start:t_end,2:3]
    nrow=len(y)
    ncol=len(y[0])
    testing=int(nrow/2)
    yval=np.zeros((nrow+future_forecast,1))

# Define the p, d and q parameters to take any value between 0 and 2
    d = q = range(0, 2)
    p = range(0,50,49)
# Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
#    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    warnings.filterwarnings("ignore") # specify to ignore warning messages
    
    min_aic=1000000
    min_param=(1,1,1)
#    min_param_seasonal=(1,1,1,12)
    for param in pdq:
#        for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
#                                                seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            if(min_aic>results.aic):
                min_param=param
#                    min_param_seasonal=param_seasonal    
                min_aic=results.aic
            
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue

    mod = sm.tsa.statespace.SARIMAX(y,
                            order=min_param,
#                            seasonal_order=min_param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
    
    results = mod.fit()
    
    print(results.summary().tables[1])
    
#    results.plot_diagnostics(figsize=(15, 12))
#    plt.show()
    
    pred = results.get_prediction(start=testing, dynamic=False)
    #pred_ci = pred.conf_int()
    #fig =Figure()
    #ax=Subplot(fig,111)
    #fig.add_subplot(ax)
    #canvas=FigureCanvas(fig)
    #ax.plot(y[200:],label='observed')
    #ax = y[200:].plot(label='observed')
    ##ax = y['1990':].plot(label='observed')
    #
    #pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
    #
    #ax.fill_between(pred_ci.index,
    #                pred_ci.iloc[:, 0],
    #                pred_ci.iloc[:, 1], color='k', alpha=.2)
    #
    #ax.set_xlabel('Time')
    #ax.set_ylabel('Segregation Index')
    #plt.legend()
    #
    #plt.show()
    y_forecasted1[0:nrow-testing,i] = pred.predicted_mean
    #y=y.shape(t_end-t_start+1,2)
    y_truth1[0:nrow-testing,i] = y[testing:,0]
    #print(y_forecasted-y_truth)
    
    # Compute the mean square error
    a=((y_forecasted1[0:nrow-testing,i] - y_truth1[0:nrow-testing,i]) ** 2).mean()
    a.shape
    mse_onestep[i] = a
    #print('The Mean Squared Error of our forecasts is {}'.format(round(mse[i], 2)))
    
    pred_dynamic = results.get_prediction(start=testing, dynamic=True, full_results=True)
    ct=(i)*2;
    pred_dynamic_ci[0:nrow-testing,ct:ct+2] = pred_dynamic.conf_int()
    
    ax = y[testing:,0]
    xval=np.zeros((testing+1,1))
    for k in range(0,testing+1):
        xval[k]=k
    fig1=plt.figure()
    plt.plot(xval,y_truth1[0:nrow-testing,i],label='DEM Calculated')
    plt.plot(xval,y_forecasted1[0:nrow-testing,i],label='ARIMA One Step ahead Forecast')
    plt.plot(xval,pred_dynamic.predicted_mean,label='ARIMA Dynamic Forecast')
    plt.xlabel('Time')
    plt.ylabel('Segregation Index')
    plt.legend()
    plt.show()
    #ax = y[testing:,0].plot(label='observed', figsize=(20, 15))
    #pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
    
    #ax.fill_between(pred_dynamic_ci.index,
    #                pred_dynamic_ci.iloc[:, 0],
    #                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
    #
    #ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1],
    #                 alpha=.1, zorder=-1)
    #
#    ax.set_xlabel('Time')
#    ax.set_ylabel('Segregation Index')
    #
    #plt.legend()
    #plt.show()
    
    # Extract the predicted and true values of our time series [DYNAMIC]
    y_forecasted[0:nrow-testing,i] = pred_dynamic.predicted_mean
    y_truth[0:nrow-testing,i] = y[testing:,0]
    
    # Compute the mean square error
    mse_dynamic[i] = ((y_forecasted[0:nrow-testing,i] - y_truth[0:nrow-testing,i]) ** 2).mean()
    #print('The Mean Squared Error of our forecasts is {}'.format(round(mse[i], 2)))
    
    # Get forecast of the next --- time steps ahead in future
    pred_uc = results.get_forecast(steps=future_forecast)
    # Get confidence intervals of forecasts
    #pred_ci= pred_uc.conf_int()
    A=pred_uc.conf_int()
    A=np.asanyarray(A)
    predf_ci[0:future_forecast,ct:ct+2]= A
#    print(y_truth.shape)
#    print(y_forecasted)
#    
#    workbook=xlsxwriter.Workbook('Prediction.xlsx')
#    worksheet = workbook.add_worksheet()
#    worksheet.write_number(y_truth)
    #worksheet.write('B1:B1000',y_forecasted)
    
    ax = y
#    bx=(np.array(predf_ci[:future_forecast,ct])+np.array(predf_ci[:future_forecast,ct+1]))/2
    bx=pred_uc.predicted_mean
    xval0=np.zeros((nrow,1))
    xval1=np.zeros((future_forecast+nrow,1))
    for k in range(0,nrow):
        xval0[k]=k
        xval1[k]=k
#    for k in range(0,nrow):
#        yval[k]=y[k]
    for k in range(nrow,future_forecast+nrow):
        if(bx[k-nrow]>1):
            xval1[k]=k
            yval[k]=bx[k-nrow]    
        else:
            xval1=xval1[0:k]
            yval=yval[0:k]
            k=future_forecast+nrow
            
    fig2=plt.figure()
    plt.plot(xval0,ax,label='observed')
    plt.plot(xval1[nrow:future_forecast],yval[nrow:future_forecast],label='Forecast')
#    plt.plot(pred_uc.predicted_mean,label='Forecast')
    #ax.fill_between(pred_ci.index,
    #                pred_ci.iloc[:, 0],
    #                pred_ci.iloc[:, 1], color='k', alpha=.25)
    #ax.set_xlabel('Time')
    #ax.set_ylabel('Segregation Index')
    #
    plt.xlabel('Time')
    plt.ylabel('Segregation Index')
    plt.legend()
    plt.show()
    
    
