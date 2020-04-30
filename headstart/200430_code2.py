#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


# In[2]:


data=pd.read_csv('data/headstart.csv')
len_before=len(data.index)
data=data.dropna(how='any',subset=['povrate60','mort_age59_related_postHS','census1960_pop'])
len_after=len(data.index)
print('Dropped',len_before-len_after,'lines that contain NaN.')


# In[3]:


# t is the threshold, k is a kernel function [0,1]->R, and b is the bandwidth
# Currently supports only 1D array
# Regression from left of t
def regr_sided(X,Y,t,k,b):
    # Dispose points outside bandwidth
    YL=Y[np.logical_and(t-b<X,X<t)]
    XL=X[np.logical_and(t-b<X,X<t)]
    
    # Calculate weights
    get_weight=np.vectorize(lambda x:k((np.abs((x-t)))/b))
    WL=get_weight(XL)

    # Reshape for lib use
    # Every data has a single feature
    XL=XL.values.reshape(-1,1)
    YL=YL.values.reshape(-1,1)
    
    # Create linear regression object
    regrL = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regrL.fit(XL, YL, sample_weight=WL)
    
    return regrL


# In[4]:


def regr_symmetric(X,Y,t,k,b):
    # Dispose points outside bandwidth
    XL=X[np.logical_and(t-b<X,X<t)]
    YL=Y[np.logical_and(t-b<X,X<t)]
    XR=X[np.logical_and(t<X,X<t+b)]
    YR=Y[np.logical_and(t<X,X<t+b)]
    
    # Calculate weights    
    get_weight=np.vectorize(lambda x:k(np.abs((x-threshold))/b))
    
    # Print weight
    print('Weights:')
    y=np.linspace(0,k(0),1000)
    x=t+0*y
    plt.plot(x,y,color='blue',linewidth=1)
    plt.plot(XL, get_weight(XL), color='purple', linewidth=2)
    plt.plot(XR, get_weight(XR), color='purple', linewidth=2)
    plt.show()
    
    regrL=regr_sided(X,Y,t,k,b)
    regrR=regr_sided(-X,Y,-t,k,b)
    
    # Make predictions using the testing set
    XL=XL.values.reshape(-1,1)
    XR=XR.values.reshape(-1,1)
    YL_pred = regrL.predict(XL)
    YR_pred = regrR.predict(np.negative(XR))
    EY_L = regrL.predict([[t]])[0][0]
    EY_R = regrR.predict(np.negative([[t]]))[0][0]
    
    # Output
    plt.scatter(XL, YL, s=0.2, color='black')
    plt.plot(XL, YL_pred, color='red', linewidth=2)
    plt.scatter(XR, YR, s=0.2, color='black')
    plt.plot(XR, YR_pred, color='red', linewidth=2)
    # Plot separating line
    y=np.linspace(0,max(EY_L,EY_R),1000)
    x=t+0*y
    plt.plot(x,y,color='blue',linewidth=1)
    plt.show()
    print('E[Y-]',EY_L)
    print('E[Y+]',EY_R)
    print('Estimated effect',EY_R-EY_L)
    
    return (regrL,regrR)


# In[5]:


threshold=59.1984
def ker_tri(x):
    return 1-x
regr_symmetric(data['povrate60'],
               data['mort_age59_related_postHS'],
               threshold,
               ker_tri,
               3)


# In[6]:


threshold=59.1984
def ker_rect(x):
    return 1 # Note that the constant here will not affect the result
regr_symmetric(data['povrate60'],
               data['mort_age59_related_postHS'],
               threshold,
               ker_rect,
               3)


# In[24]:


def get_interval(X,Y,a,b):
    return X[np.logical_and(a<X,X<b)],Y[np.logical_and(a<X,X<b)]

def validator3(X,Y,k,b):
    n=100
    err_sum=0
    l=np.amin(X)
    r=np.amax(X)
    XS=get_interval(X,Y,l+b,r)[0]
    for test_case in range(n):
        #select a random node
        i=XS.index[np.random.choice(len(XS.index))]
        x,y=X[i],Y[i]
        regr_model=regr_sided(X,Y,x,k,b)
        ey=regr_model.predict([[x]])[0][0]
        err_sum+=(ey-y)*(ey-y)
    return err_sum/n

validator3(data['povrate60'],data['mort_age59_related_postHS'],ker_rect,3)


# In[26]:


# t is the threshold
def cv_test_bandwidth(X,Y,t,k,vald):
    for i in np.linspace(0.5,40,100):
        err=(vald(X,Y,k,i)+vald(-X,Y,k,i))/2
        plt.scatter(i, err, s=0.2, color='blue')
    plt.show()

cv_test_bandwidth(data['povrate60'],data['mort_age59_related_postHS'],threshold,ker_tri,validator3)

