# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:08:42 2017

@author: kumarpx21
"""

"""
==================================
Demo of the basics of violin plots
==================================

Violin plots are similar to histograms and box plots in that they show
an abstract representation of the probability distribution of the
sample. Rather than showing counts of data points that fall into bins
or order statistics, violin plots use kernel density estimation (KDE) to
compute an empirical distribution of the sample. That computation
is controlled by several parameters. This example demonstrates how to
modify the number of points at which the KDE is evaluated (``points``)
and how to modify the band-width of the KDE (``bw_method``).

For more information on violin plots and KDE, the scikit-learn docs
have a great section: http://scikit-learn.org/stable/modules/density.html
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

#mat_contents=sio.loadmat('C:/Users/kumarpx21/Desktop/Intern/Codes/Simulations/PSL/input.mat')
mat_contents=sio.loadmat('/home/kumarpx21/Intern_Files/Simulations/PSL/input.mat')
Input=mat_contents['Data']
Input[:,10]=0
Input=np.matrix(Input)

ct=-1
Data=np.zeros((len(Input),9))
Response=np.zeros((len(Input),1))
R1=mat_contents['Features']
R2=mat_contents['Response']
complete_sim=R1[:,1] # Complete/Incomplete simulations
(m,n)=R1.shape
for i in range(0,len(complete_sim)):
    if(complete_sim[i]==1 and R1[i,2]==0):
        ct=ct+1;
        Data[ct,:]=R1[i,3:12] # Features
        Response[ct]=R2[i,6] # Mixing Time
Data=Data[0:ct+1,:]
Response=Response[0:ct+1]

# fake data
fs = 15 # fontsize
#pos = [1, 2, 4, 5, 7, 8]
pos=1
#data = [np.random.normal(0, std, size=100) for std in pos]

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 14))
data=Data[:,0]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[0, 1].violinplot(data, showmeans=True)
axes[0, 1].set_title('Particle Size ($\mu$m)', fontsize=fs,fontweight='bold')
axes[0,1].axis([0,2,0,6000])
axes[0,1].set_color('green')
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')

data=Data[:,1]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[0, 0].violinplot(data, showmeans=True)
axes[0, 0].set_title('Cohesive Energy Density ($J/m^3$)', fontsize=fs,fontweight='bold')
axes[0, 0].axis([0,2,0,100000])
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')

##
data=Data[:,2]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[1, 0].violinplot(data, showmeans=True)
axes[1, 0].set_title('Tangential Friction', fontsize=fs,fontweight='bold')
axes[1, 0].axis([0,2,0,1])
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')

#
data=Data[:,3]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[1, 1].violinplot(data, showmeans=True)
axes[1, 1].set_title('Coefficient of Restitution', fontsize=fs,fontweight='bold')
axes[1, 1].axis([0,2,0,0.60])
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')

##
data=Data[:,4]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[2, 0].violinplot(data, showmeans=True)
axes[2, 0].set_title('Youngs Modulus ($N/m^2$)', fontsize=fs,fontweight='bold')
axes[2, 0].axis([0,2,0,100000000])
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')


##
data=Data[:,5]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[2, 1].violinplot(data, showmeans=True)
axes[2, 1].set_title('Particle Density ($kg/m^3$)', fontsize=fs,fontweight='bold')
axes[2, 1].axis([0,2,0,2000])
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')


#
data=Data[:,6]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[3, 0].violinplot(data, showmeans=True)
axes[3, 0].set_title('Number of Particles', fontsize=fs,fontweight='bold')
axes[3, 0].axis([0,2,0,2000000])
for ax in axes.flatten():
    ax.set_xticklabels([])
parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')

#
data=Data[:,7]
data=np.transpose(data)
data=np.ndarray.tolist(data)
axes[3, 1].violinplot(data, showmeans=True)
axes[3, 1].set_title('Impeller RPM', fontsize=fs,fontweight='bold')
axes[3, 1].axis([0,2,0,15],fontweight='bold')

for ax in axes.flatten():
    ax.set_xticklabels([])
#

parts = ax.violinplot(data, showmeans=True, showmedians=False,showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')
#    pc.set_alpha(1)

##fig.suptitle("Violin Plotting Examples")
#fig.subplots_adjust(hspace=0.4)
plt.show()
fig.savefig('/home/kumarpx21/Intern_Files/Simulations/PSL/plots/input_features_14sep18.pdf',facecolor='white', edgecolor='white')
