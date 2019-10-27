

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



## load data
trainSet = pd.read_csv("clevelandtrain.csv")
testSet = pd.read_csv("clevelandtest.csv")

xtrain = (trainSet.drop(["heartdisease::category|0|1"], axis=1)).iloc[:,:].values  # (152, 13)
ytrain = trainSet["heartdisease::category|0|1"].iloc[:].values                     # (152,)

xtest = (testSet.drop(["heartdisease::category|0|1"], axis=1)).iloc[:,:].values    # (145, 13)
ytest = testSet["heartdisease::category|0|1"].iloc[:].values                       # (145,)

# print("the first 4 raw data is:\n" ,xtrain[0:3,:])

## standardize the data
from sklearn.preprocessing import StandardScaler
xtrain = StandardScaler().fit_transform(xtrain)
xtest = StandardScaler().fit_transform(xtest)

# print("the first 4 standardized data is: \n", xtrain[0:3,:])


## project the data into 2 dimensions, where new components are just the two main dimensions of variation.
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(xtrain)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, trainSet[['heartdisease::category|0|1']]], axis = 1)
# print("the first 4 data in pca spacce: \n", finalDf.head(4))

## Visualize 2D Projection
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['heartdisease::category|0|1'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

var_ratio2 = pca2.explained_variance_ratio_
var2 = pca2.explained_variance_
cmpnt2 = pca2.components_

print(var_ratio2)

pca = PCA(n_components=13)
principalComponents = pca.fit_transform(xtrain)

## how much information(variance) we get
var_ratio = pca.explained_variance_ratio_
var = pca.explained_variance_
cmpnt = pca.components_



print(var_ratio)
print(var)
print(cmpnt)
import matplotlib.pyplot as plt
import plotly.plotly as py
py.sign_in(username='Shaohua_Zhang', api_key='tnr36EaVoikjm6IaPkRs')
# import plotly
# plotly.tools.set_credentials_file(username='Shaohua_Zhang', api_key='tnr36EaVoikjm6IaPkRs')

trace1 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,14)],
    y=var_ratio,
    name='Individual'
)

trace2 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,14)],
    y=np.cumsum(var_ratio),
    name='Cumulative'
)

data = [trace1, trace2]
layout=dict(
    title='Explained variance by different principal components',
    yaxis=dict(
        title='Explained variance in percent'
    )
)

fig = dict(data=data, layout=layout)
py.plot(fig, filename='exploratory-vis-histogram')