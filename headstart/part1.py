import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def bias1(X,Y,t,k,b):
    X0=X[np.logical_and(t-b<X,X<t)]
    X1=X0-t                             #take threshold to x=0
    Y1=Y[np.logical_and(t-b<X,X<t)]
    get_weight=np.vectorize(lambda x:k((np.abs(x))/b))
    WL=get_weight(X1)                   
    XW=X1*WL                            #weighted X1
    XXW=X1**2*WL                        #weighted X1^2
    YW=Y1*WL                            #weighted Y1
    #averages
    Waver=np.sum(WL)
    XXaver=np.sum(XXW)/Waver
    Xaver=np.sum(XW)/Waver
    Yaver=np.sum(YW)/Waver
    regr = linear_model.LinearRegression()
    regr.fit(X1, Y1, sample_weight=WL)
    a=regr.coef_                        #find the slope of regression
    f1=WL*(a*X1-a*Xaver-Y1+Yaver)
    f2=WL*(X1-XXaver)**2
    n=np.size(X)
    F=XXaver*np.sum(f1)/((n-1)*np.sum(f2))
    return F

