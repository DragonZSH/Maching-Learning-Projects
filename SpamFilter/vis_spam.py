
from valsplit import valsplit
from linearmodel import linearmodel
from scipy import io
import numpy as np

def vis_spam(w):
#         b=np.loadtxt('data/data_train/index',dtype=str)
#         label = b[:,0]
#         path = b[:,1]
#         
#         Y=np.zeros((1,len(label)))
#         for i in range(len(label)):
#             if(label[i]=='spam'):
#                 Y[:,i]=1
#             else:
#                 Y[:,i]=-1
                
        data = io.loadmat('data_train.mat')
        X = data['X']
        Y = data['Y']
        
        # split the data:
        xTr,xTv,yTr,yTv = valsplit(X,Y)
        
        
        correct=[]
        for i in range(len(yTv[0])): 
            p=linearmodel(w,xTv[:,i])
            
            if p>0:
                    pred='SPAM'
            else: 
                    pred='GOOD'
            
            if yTv[:,i]==1:
                truth='SPAM'
            else:
                truth='GOOD'
            if (yTv[:,i] != np.sign(p)):
                correct.append(0)    
                print('Wrong: %s TRUTH: %s \n' % (pred,truth))
                Accuracy = sum(correct)*100.0/len(correct)
                print('Accuracy %.2f\n'%Accuracy)
            else:
                correct.append(1)
                print('Correct: %s TRUTH: %s \n' % (pred,truth))