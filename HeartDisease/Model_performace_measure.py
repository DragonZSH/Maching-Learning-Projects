

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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, \
    roc_curve





## load data
from plot_confusion_matrix import plot_confusion_matrix

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


print("-----------------------------------------------------------------")


class_names = np.array([1,0])


svcRBF = SVC(C=300.0,gamma=0.0001,kernel='rbf',probability=True)
svcRBF.fit(xTrain,yTrain)
svcRBFScore = svcRBF.score(xTest, yTest)
prediction_svcRBF = svcRBF.predict(xTest)



# # Plot non-normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_svcRBF, classes=class_names, title='Confusion matrix svcRBF, without normalization')
#
# # Plot normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_svcRBF, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix svcRBF')
#
# plt.show()
#
#
f1 = f1_score(yTest, prediction_svcRBF)
acc1 = accuracy_score(yTest, prediction_svcRBF)
rec1 = recall_score(yTest, prediction_svcRBF)
pre1 = precision_score(yTest, prediction_svcRBF)
auc1 = roc_auc_score(yTest, prediction_svcRBF)
print(f1)
print(acc1)
print(rec1)
print(pre1)
print(auc1)
print("-----------------------------------------------------------------")




svcPoly = SVC(C=1.0,degree = 8.666666,coef0=1.0,gamma = 'scale',max_iter=-1,kernel='poly',probability=True)
svcPoly.fit(xTrain,yTrain)
svcPolyScore = svcPoly.score(xTest, yTest)
prediction_svcPLOY = svcPoly.predict(xTest)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_svcPLOY, classes=class_names, title='Confusion matrix svcPLOY, without normalization')
#
# # Plot normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_svcPLOY, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix svcPLOY')
#
# plt.show()
#
f2 = f1_score(yTest, prediction_svcPLOY)
acc2 = accuracy_score(yTest, prediction_svcPLOY)
rec2 = recall_score(yTest, prediction_svcPLOY)
pre2 = precision_score(yTest, prediction_svcPLOY)
auc2 = roc_auc_score(yTest, prediction_svcPLOY)
print(f2)
print(acc2)
print(rec2)
print(pre2)
print(auc2)
print("-----------------------------------------------------------------")



decisonTree = tree.DecisionTreeClassifier()
decisonTreeBagging = BaggingClassifier(decisonTree,max_samples=0.7, max_features=1.0)
decisonTreeBagging.fit(xTrain,yTrain)
Bagging_score = decisonTreeBagging.score(xTest,yTest)

prediction_Bagging = decisonTreeBagging.predict(xTest)



# # Plot non-normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_Bagging, classes=class_names, title='Confusion matrix decisonTreeBagging, without normalization')
#
# # Plot normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_Bagging, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix decisonTreeBagging')
#
# plt.show()
#
f3 = f1_score(yTest, prediction_Bagging)
acc3 = accuracy_score(yTest, prediction_Bagging)
rec3 = recall_score(yTest, prediction_Bagging)
pre3 = precision_score(yTest, prediction_Bagging)
auc3 = roc_auc_score(yTest, prediction_Bagging)
print(f3)
print(acc3)
print(rec3)
print(pre3)
print(auc3)
print("-----------------------------------------------------------------")



decisonTreeAda = AdaBoostClassifier(decisonTree,n_estimators=10,random_state=np.random.RandomState(1))
decisonTreeAda.fit(xTrain,yTrain)
AdaBoost_score = decisonTreeAda.score(xTest,yTest)

prediction_AdaBoost = decisonTreeAda.predict(xTest)


# # Plot non-normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_AdaBoost, classes=class_names, title='Confusion matrix AdaBoost, without normalization')
#
# # Plot normalized confusion matrix
# plot_confusion_matrix(yTest, prediction_AdaBoost, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix AdaBoost')
#
# plt.show()
#
#
f4 = f1_score(yTest, prediction_AdaBoost)
acc4 = accuracy_score(yTest, prediction_AdaBoost)
rec4 = recall_score(yTest, prediction_AdaBoost)
pre4 = precision_score(yTest, prediction_AdaBoost)
auc4 = roc_auc_score(yTest, prediction_AdaBoost)
print(f4)
print(acc4)
print(rec4)
print(pre4)
print(auc4)
print("-----------------------------------------------------------------")





## plot roc curve

prediction_prob_svmRBF = svcRBF.predict_proba(xTest)[:, 1]
prediction_prob_svmPOLY = svcPoly.predict_proba(xTest)[:, 1]
prediction_prob_Bagging = decisonTreeBagging.predict_proba(xTest)[:, 1]
prediction_prob_Adaboost = decisonTreeAda.predict_proba(xTest)[:, 1]

fpr_svmRBF, tpr_svmRBF, _ = roc_curve(yTest, prediction_prob_svmRBF)
fpr_svmPOLY, tpr_svmPOLY, _ = roc_curve(yTest, prediction_prob_svmPOLY)
fpr_Bagging, tpr_Bagging, _ = roc_curve(yTest, prediction_prob_Bagging)
fpr_Adaboost, tpr_Adaboost, _ = roc_curve(yTest, prediction_prob_Adaboost)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_svmRBF, tpr_svmRBF, label='SVM-RBF')
plt.plot(fpr_svmPOLY, tpr_svmPOLY, label='SVM-POLY')
plt.plot(fpr_Bagging, tpr_Bagging, label='Bagging')
plt.plot(fpr_Adaboost, tpr_Adaboost, label='Adaboost')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()








