from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit

from scipy import io
import numpy as np

from checkgradLogistic import checkgradLogistic
from checkgradHingeAndRidge import checkgradHingeAndRidge

from ridge import ridge
from hinge import hinge
from logistic import logistic

# load the data:
data = io.loadmat('data/data_train.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)

# uncomment checkgrad for a function to check that gradients are implemented correctly
# error should be very low (under 1e-8 for defualt parameters)

small_step = 1e-5
feature_vector = np.zeros((xTr.shape[0],1))
lambdaa = 10

ridge_error = checkgradHingeAndRidge(ridge, feature_vector, small_step, xTr, yTr, lambdaa)
print("Ridge error is", ridge_error)

hinge_error = checkgradHingeAndRidge(hinge, feature_vector, small_step, xTr, yTr, lambdaa)
print("Hinge error is", hinge_error)

logistic_error = checkgradLogistic(logistic, feature_vector, small_step, xTr, yTr)
print("Logistic error is", logistic_error)

# train spam filter with parameters in trainspamfilter.py
        
w_trained = trainspamfilter(xTr,yTr)
spamfilter(xTv,yTv,w_trained,0.1)












