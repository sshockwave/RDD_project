import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def ker_tri(x):
    return 2*(1-x)
def ker_rect(x):
    return 1
def ker_quad1(x):
    return ((1-x)**2)
def ker_quad2(x):
    return(1-x**2)


n1=80
n2=10
m=20
X=np.random.rand(n1)*10-5+59 #can change the distribution of X
for i in range(1,m):
    A=59-0.2*i-np.random.rand(n2)*5
    X=np.hstack((X,A))
def actual_f(x):
    return 0.2*((x-59)**2)+2
Y=np.vectorize(actual_f)(X)+1*np.random.rand(np.size(X))+1.2*np.random.rand(np.size(X))-1.1

def regr_sided(X,Y,t,k,b):
    # Dispose points outside bandwidth
    YL=Y[np.logical_and(t-b<X,X<t)]
    XL=X[np.logical_and(t-b<X,X<t)]
    
    # Calculate weights
    get_weight=np.vectorize(lambda x:k((np.abs((x-t)))/b))
    WL=get_weight(XL)

    # Reshape for lib use
    # Every data has a single feature
    XL=XL.reshape(-1,1)
    YL=YL.reshape(-1,1)
    
    # Create linear regression object
    regrL = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regrL.fit(XL, YL, sample_weight=WL)
    
    return regrL


def bias(X,Y,t,k,i):
    final_regr=regr_sided(X,Y,t,k,i)
    true_err=(final_regr.predict([[t]])[0][0]-actual_f(t))**2
    return true_err

def test_kernel(X,Y,t,k1,k2,k3):
    for i in np.exp(np.linspace(-0.5,1.7,100)):
        err1=bias(X,Y,t,k1,i)
        err2=bias(X,Y,t,k2,i)
        err3=bias(X,Y,t,k3,i)
        #err2=bias2(X,Y,t,k,i,regr2)
        plt.scatter(i, err1, s=0.2, color='blue')
        plt.scatter(i, err2, s=0.2, color='green')
        plt.scatter(i, err3, s=0.2, color='orange')
    plt.show()
    

test_kernel(X,Y,59,ker_quad1,ker_tri,ker_quad2)