import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data=pd.read_excel('D:\OneDrive\课程\大一下\因果与统计学习\My own project\project数据\dataoperation.xlsx')

dataL=data[data['povrate60']<59.19]
dataR=data[data['povrate60']>59.19]

# Load the diabetes dataset
XL = dataL.loc[:,['povrate60']]
YL = dataL.loc[:,['mort_age59_related_postHS']]
XR = dataR.loc[:,['povrate60']]
YR = dataR.loc[:,['mort_age59_related_postHS']]

# Use only one feature
#diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets



# Split the targets into training/testing sets


#kernel is set here
#ker=X

# Create linear regression object
regrL = linear_model.LinearRegression()
regrR = linear_model.LinearRegression()
# Train the model using the training sets
regrL.fit(XL, YL)#attach kernel
regrR.fit(XR, YR)#attach kernel
# Make predictions using the testing set
YL_pred = regrL.predict(XL)
YR_pred = regrR.predict(XR)
'''
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y, diabetes_y_pred))
'''
# Plot outputs
plt.scatter(XL, YL, s=0.2, color='black')
plt.plot(XL, YL_pred, color='red', linewidth=2)
plt.scatter(XR, YR, s=0.2, color='black')
plt.plot(XR, YR_pred, color='red', linewidth=2)

y=np.linspace(0,30,1000)
x=59.19+0*y
plt.plot(x,y,color='blue',linewidth=1)

plt.xticks(())
plt.yticks(())

plt.show()
