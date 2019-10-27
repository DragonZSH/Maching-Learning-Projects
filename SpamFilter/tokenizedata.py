import subprocess
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse import dok_matrix
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import normalize

def tokenizedata(directory='data/data_train',output='data_train'):
    
        PYTHONHASHSEED = 0
        
        HASHBUCKETS=2**10
        ind=[x.split() for x in open(directory+'/index').read().split('\n') if len(x)>0] # read in index file
        codebook={}
        
        spamdata = "spamdata.csv"
        spamdatafile = open(spamdata, "w")
        
        num_features = HASHBUCKETS
        num_examples = 5000
        M = dok_matrix((num_features, num_examples), dtype=np.int)
        
        # build codebook
        file_number = 0
        for (num,(label,fn)) in enumerate(ind):
        		# the next command loads in the data, replaces returns with blanks, splits it into words and hashes the words into integers
            email=map(lambda e: abs(e.__hash__()) % HASHBUCKETS,open(directory+'/'+fn, encoding = "ISO-8859-1").read().replace('\n',' ').split())
        		# write out "emailnumber,hashedword" for all words
            M[list(email), file_number] = 1 

            file_number += 1
            
        
        X = normalize(M, norm='l1', axis=0)    
        
        # Load in labels (do not change this part of the code)
        if(directory=='data/data_test'):
            b=np.loadtxt('data/data_test/index',dtype=str)
        else:
            b=np.loadtxt('data/data_train/index',dtype=str)
        label = b[:,0]
        

        Y=np.zeros((1,len(label)))
        for i in range(len(label)):
            if(label[i]=='spam'):
                Y[:,i]=1
            else:
                Y[:,i]=-1
 
        if(output=='data_test'):  
            sio.savemat('data/data_test.mat', {'X': X,'Y': Y})
        else:
            sio.savemat('data/data_train.mat', {'X': X,'Y': Y})

if __name__ == '__main__':
    tokenizedata()




