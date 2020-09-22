from IFC2F import IFC2F
import numpy as np
import scipy.io

'''
Step 1. Load dataset:
    Xtrain: Train data 
    Ytrain: Train labels
    Xtest: test data
    Ytest: test labels
'''

mat = scipy.io.loadmat('Datasets/Gene/MLL_data_label_all.mat')
   
Xtrain = np.array(mat['train_data'],dtype = None)
Ytrain = mat['train_label'].astype(int)[:,0]
Xtest = np.array(mat['test_data'],dtype =None)
Ytest = mat['test_label'].astype(int)[:,0]

dataset = {'Xtrain': Xtrain, 'Ytrain': Ytrain, 'Xtest':Xtest, 'Ytest':Ytest}

'''
Step 2. Define configuration parameters:
    feat_cost: feature evaluation cost
    bins: number of bins conisdered when quantizing the feature space
    beta: number of reachable belief points
'''
config = {'feat_cost': 0.01, 'bins':4, 'beta': 100}

'''
Step 3. Initiate an instance of IFC2F using config file and call run function
'''
clf = IFC2F(config)
clf.run(dataset)

'''
Step 4. Print classification summary report
'''
print("Classification Report: "+str(clf.summary)+'\n')
