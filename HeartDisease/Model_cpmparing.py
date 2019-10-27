

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score





## load data
trainSet = pd.read_csv("clevelandtrain.csv")
testSet = pd.read_csv("clevelandtest.csv")

xtrain = (trainSet.drop(["heartdisease::category|0|1"], axis=1)).iloc[:,:].values  # (152, 13)
ytrain = trainSet["heartdisease::category|0|1"].iloc[:].values                     # (152,)

xtest = (testSet.drop(["heartdisease::category|0|1"], axis=1)).iloc[:,:].values    # (145, 13)
ytest = testSet["heartdisease::category|0|1"].iloc[:].values                       # (145,)



## data preprocessing

 # without: one-hot-encoder:

# xTrain = xtrain
# yTrain = ytrain
# xTest = xtest
# yTest = ytest

  # one-hot-encoder: #9 (cp), #19 (restecg),  #41 (slope), #51 (thal)

xtrain_pre = trainSet.drop(["cp", "restecg", "slope", "thal", "heartdisease::category|0|1"], axis=1).iloc[:,:].values # (152, 9)
xtrain_cp = trainSet["cp"].iloc[:].values
xtrain_restecg = trainSet["restecg"].iloc[:].values
xtrain_slope = trainSet["slope"].iloc[:].values
xtrain_thal = trainSet["thal"].iloc[:].values

ohe1 = OneHotEncoder(sparse = False,categories='auto',handle_unknown='ignore')
ohe2 = OneHotEncoder(sparse = False,categories='auto',handle_unknown='ignore')
ohe3 = OneHotEncoder(sparse = False,categories='auto',handle_unknown='ignore')
ohe4 = OneHotEncoder(sparse = False,categories='auto',handle_unknown='ignore')

xtrain_cp = ohe1.fit_transform(xtrain_cp.reshape(-1,1))                    # (152, 4)
xtrain_restecg = ohe2.fit_transform(xtrain_restecg.reshape(-1,1))          # (152, 3)
xtrain_slope = ohe3.fit_transform(xtrain_slope.reshape(-1,1))              # (152, 3)
xtrain_thal = ohe4.fit_transform(xtrain_thal.reshape(-1,1))                # (152, 3)


xTrain = np.hstack((xtrain_pre, xtrain_cp, xtrain_restecg, xtrain_slope, xtrain_thal))   # (152, 22)
yTrain = ytrain                                                                          # (152,)



xtest_pre = testSet.drop(["cp", "restecg", "slope", "thal", "heartdisease::category|0|1"], axis=1).iloc[:,:].values   # (145, 9)
xtest_cp = testSet["cp"].iloc[:].values
xtest_restecg = testSet["restecg"].iloc[:].values
xtest_slope = testSet["slope"].iloc[:].values
xtest_thal = testSet["thal"].iloc[:].values

xtest_cp = ohe1.transform(xtest_cp.reshape(-1,1))                 # (145, 4)
xtest_restecg = ohe2.transform(xtest_restecg.reshape(-1,1))       # (145, 3)
xtest_slope = ohe3.transform(xtest_slope.reshape(-1,1))           # (145, 3)
xtest_thal = ohe4.transform(xtest_thal.reshape(-1,1))             # (145, 3)

xTest = np.hstack((xtest_pre, xtest_cp, xtest_restecg, xtest_slope, xtest_thal))   # (145, 22)
yTest = ytest                                                                      # (145,)




svcRBF = SVC(C=300.0,gamma=0.0001,kernel='rbf',probability=True)
svcRBF.fit(xTrain,yTrain)
svcRBFScore = svcRBF.score(xTest, yTest) # test accuracy
print("the test score of svcRBFScore"+str(svcRBFScore))


svcPoly = SVC(C=1.0,degree = 8.666666,coef0=1.0,gamma = 'scale',max_iter=-1,kernel='poly',probability=True)
svcPoly.fit(xTrain,yTrain)
svcPolyScore = svcPoly.score(xTest, yTest) # test accuracy
print("the test score of svcPolyScore"+str(svcPolyScore))


decisonTree = tree.DecisionTreeClassifier()
decisonTreeBagging = BaggingClassifier(decisonTree,max_samples=0.7, max_features=1.0)
decisonTreeAda = AdaBoostClassifier(decisonTree,n_estimators=10,random_state=np.random.RandomState(1))

decisonTreeBagging.fit(xTrain,yTrain)
decisonTreeAda.fit(xTrain,yTrain)
Bagging_score = decisonTreeBagging.score(xTest,yTest)
AdaBoost_score = decisonTreeAda.score(xTest,yTest)

print("the test score of Bagging:", Bagging_score)
print("the test score of Adaboost:", AdaBoost_score)















