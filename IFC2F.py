import numpy as np
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from scipy import sparse, linalg
from pyitlib import discrete_random_variable as drv
from typing import Set, TypeVar
import warnings, time 
warnings.simplefilter(action='ignore', category=FutureWarning)    
warnings.simplefilter(action='ignore', category=RuntimeWarning)  

class IFC2F(object):
    """
    IFC2F Classs Object
    Initializing Attributes:
        feat_cost: feature evaluation cost
        bins: number of bins conisdered when quantizing the feature space
        beta: number of reachable belief points
    Functions: 
        run: main function of IFC2F object which perform model training and testing. input must be a dataset
        fit: train the model
        predict: perform instance-wise dynamic joint feature selection and classification on test dataset
        preprocess: preprocessing steps; get feature distributions, class prior distribution, learn Bayesian network 
    """   
    def __init__(self, config):
        self.feat_cost = config['feat_cost']
        self.bins      = config['bins']
        self.beta      = config['beta']
        
    def run(self, data):    
        # preprocessing
        self.preprocess(data['Xtrain'], data['Ytrain'])     
        
        # training
        train_time = self.fit()
        
        # joint feature selection and classification
        predictions, n_feat, fs_and_c_time = self.predict(data['Xtest'])
        
        # classification report
        Ytest = np.array([self.C.index(x) for x in data['Ytest']])  
        self.summary = {'accuracy': accuracy_score(Ytest, predictions), 'avg_feat': np.mean(n_feat),
                       'std_feat': np.std(n_feat),'training_time': train_time, 'testing_time': fs_and_c_time}
               
    def fit(self):
        # training IFC2F utilzing an extension of Perceus Algorithm 
        
        stages = len(self.Marginals)+1
        self.J = [[] for _ in range(stages)]
        self.Opt_A = [[] for _ in range(stages)]   
        
        # initialize value function with alpha vectors from final stage, i.e., for stopping action
        self.J[stages -1] = self.Q[:].tolist()
        self.Opt_A[stages -1]  = [0 for x in range(self.L)][:]

        start = time.time() # starting time

        for i in range(stages-2,-1,-1):

            B_bar = self.B[i,:]  # set to be improved       
            cond_in = self.Cond_ind[i,:]

            while len(B_bar)>0:
                rand  = np.random.randint(0,len(B_bar))
                b = B_bar[rand]  # random belief point from i th stage

                H = self.Marginals[i,:,:,cond_in[rand]]  # observation matrix of next stage 

                [action,alpha,value] = backup(b,self.J[i+1], self.Q, self.feat_cost, self.L, self.bins, H)

                V_prev_alpha = self.J[i+1][np.argmin(np.dot(self.J[i+1],b))]


                if value < np.dot(V_prev_alpha,b):               
                    alpha_b = alpha
                    action_b = action
                else:              
                    alpha_b = V_prev_alpha
                    action_b  = self.Opt_A[i+1][np.argmin(np.dot(self.J[i+1],b))] 

                if  not self.J[i]:
                        self.J[i].append(alpha_b)            
                        self.Opt_A[i].append(action_b)
                elif alpha_b not in self.J[i]:
                        self.J[i].append(alpha_b)            
                        self.Opt_A[i].append(action_b)

                B_new  = []   
                for z in B_bar:  
                    if  np.min(np.dot(self.J[i],z)) > np.min(np.dot(self.J[i+1],z)):
                        B_new.append(z)
                B_bar = np.array(B_new[:])

        train_time = time.time() - start # training time
        return train_time
        
    def predict(self, Xtest):
        # predict test instances using IFC2F Algorithm

        Xtest = Xtest[:,self.ordering]  
        predictions = []  # predictions
        n_feat = []       # number of features used

        start = time.time() 
        for z in range(len(Xtest)):    
            obs = Xtest[z,:] #present obervation

            d = len(self.f_ordering)
            pin = self.prior
            for i,f in enumerate(self.f_ordering):
                value_f = obs[f]  # feature value   
                index = find_range(value_f,self.edges[f][1:])             
                if i == 0: # if root node
                    prob = self.CPD_r[:,index]                  
                else:        
                    cond_paren = self.Conditioned_on[i-1]
                    value_p = obs[cond_paren]                
                    index_p = find_range(value_p,self.edges[cond_paren][1:])                
                    prob = self.Marginals[i-1,:,index,index_p]

                pin = np.divide(np.multiply(pin,prob),np.dot(pin,prob))       

                if self.Opt_A[i][np.argmin(np.dot(self.J[i],pin))] ==0:
                    d = i+1
                    break

            D = np.argmin(np.dot(self.Q,pin))
            n_feat.append(d)

            predictions.append(D)  # predictions

        fs_and_c_time =  time.time() - start  # joint feature selection and classification time  
        return predictions, n_feat, fs_and_c_time


    def preprocess(self, Xtrain, Ytrain):
        
        self.C = list(set(Ytrain))       # classes
        self.L = len(self.C)             # number of classes      
        self.Q = cost_matrix(self.L)     # misclassification cost matrix

        Ytrain = np.array([self.C.index(x) for x in Ytrain]) # convert class variable into integers 
        
        self.prior = np.array([sum(Ytrain==x)/len(Ytrain) for x in range(self.L)]) # priori probabilities   

        C_F, self.ordering = mutual_info_order(Xtrain, Ytrain) # mutual information with class variable
        Xtrain = Xtrain[:,self.ordering]

        # extract highly correlated features with the class variable
        self.K = automatic_thresh(C_F) # filtered features
        
        # quantize feature space
        Xtrain, self.edges = quantize_fspace(Xtrain, self.K, self.bins)

        # compute pairwise conditional mutual information
        Mi = MI(Xtrain,Ytrain, self.prior, self.L)  

        # generate the maximum spanning tree
        graph = maximum_spanning_directed_tree(self.K, Mi)

        # create CPDs  
        self.CPD_r, CPDs = create_cpds_ML(Xtrain,Ytrain,graph,self.L,self.bins, self.K, self.prior)     

        # Markov Blanket based feature ordering
        self.f_ordering = Markov_blanket_ord(graph,self.K)  

        # find ancestors 
        ancestors = find_ancestors(self.f_ordering, graph) 

        # compute margials 
        self.Marginals, self.Conditioned_on = compute_marginals(CPDs, ancestors,self.f_ordering,self.L, self.bins)

        # get a random set B of reachable belief points for each stage  
        self.B, self.Cond_ind = reachable_beliefs(self.Marginals,self.CPD_r, self.f_ordering,
                                                  self.Conditioned_on,self.prior,self.L, self.bins, self.beta)

def cost_matrix(L):
    # misclassification cost matrix
    arr, arr[0] = np.ones(L), 0
    return linalg.toeplitz(arr)

def quantize_fspace(Xtrain,K,Bins):
    # quantize feature space
    Edges = np.zeros((K,Bins+1))
    X_new = np.zeros((np.size(Xtrain,axis=0),K))
    for i in range(K):
        min_r = np.floor(Xtrain[:,i].min())
        max_r = np.ceil(Xtrain[:,i].max())
        Edges[i] = np.linspace(min_r, max_r, num=Bins+1)
        X_new[:,i] = np.digitize(Xtrain[:,i],bins=Edges[i])       
    return X_new,Edges

def MI(Xtrain,Ytrain,P,L):
    # pairwise mutual information
    K = np.size(Xtrain,axis=1)
    mutual_info = np.zeros((K, K))
    for x in range(K):
        X1 = Xtrain[:,x].astype(int)
        for y in range(x+1,K):
            Y1 = Xtrain[:,y].astype(int)
            out = drv.information_mutual_conditional(X1,Y1,Ytrain)               
            mutual_info[x, y] = max(0,out)             
    return mutual_info

T = TypeVar('T')
def discard(s: Set[T], value: T):
    count = len(s)
    s.discard(value)
    return len(s) != count

def maximum_spanning_directed_tree(num_nodes, weighted_edges):
    # generate maximum spanning directed tree using mutual information
    adjacencies = sparse.csr_matrix(-weighted_edges)
    mst = sparse.csgraph.minimum_spanning_tree(adjacencies)
    edges = {frozenset((i, j)) for i, j in zip(*mst.nonzero())}
    parent = -np.ones(num_nodes, dtype=int)
    visited = {0}
    while edges:
        for i in range(1, num_nodes):
            for j in range(num_nodes):
                if j in visited and discard(edges, frozenset((i, j))):
                    visited.add(i)
                    parent[i] = j
    return parent

def create_cpds_ML(Xtrain,Ytrain,graph,L,Bins,K,P):
    # approximate conditional probability tables using ML estimates
    CPD_r = np.zeros((L,Bins))
    CPDs = np.zeros((K-1,L,Bins,Bins))
    BB = np.arange(0.9,Bins+1)
    Lp = 1 # laplase prior
    for j in range(L):    
        Filt = Xtrain[Ytrain == j]    
        for i, parent in enumerate(graph):        
            if parent <0:  #creating the cpd for root node
                CPD = np.histogram(Filt[:,i], bins=BB)[0]
                CPD_r[j,:] = (CPD+Lp)/(sum(CPD) + Bins*Lp)
            else:        #creating cpds for all other nodes
                CPD  = np.histogram2d(Filt[:,i],Filt[:,parent], bins = [BB,BB])[0]
                CPDs[i-1,j,:] = (CPD+Lp)/(np.sum(CPD,axis=0)+Bins*Lp)  
                
    return CPD_r,CPDs

def Markov_blanket_ord(graph,K):  
    # Markov blanket based ordering
    f_ordering = []
    all_nodes = [x for x in range(K)]
    
    while all_nodes:      
        cur_node = all_nodes.pop(0)
        f_ordering.append(cur_node) #add highest mutual information node       
        
        if graph[cur_node] in all_nodes:  #remove parent
            all_nodes.remove(graph[cur_node]) 
            
        children = np.where(graph==cur_node)[0] 
        all_nodes = [x for x in all_nodes if x not in children]    # remove children    
        # note: here I didnt remove the parents of children because TAN structure has only one parent.        
    return f_ordering

def mutual_info_order(Xtrain,Ytrain):
    # computing mutual information with class variable
    C_F = np.array([adjusted_mutual_info_score(Xtrain[:,x], Ytrain)
                    for x in range(np.size(Xtrain,axis=1))])
    
    ordering = sorted(range(len(C_F)), key=lambda k:C_F[k],reverse=True) # order with respect to mutual information
    return C_F, ordering

def automatic_thresh(C_F):  
    # automatic threshold on mutual information
    thresh =0.8
    while sum(C_F>thresh) < min(len(C_F)/2,15):        
        thresh = thresh/2
        if thresh ==0:
            raise ValueError('Training dataset is Trash....')           
    return sum(C_F>thresh)    

def compute_marginals(CPDs, ancestors, f_ordering, L, Bins):
    # approximate second-order dependency marginals
    K_new = len(f_ordering)
    Marginals = np.zeros((K_new-1,L,Bins,Bins))
    Conditioned_on = np.zeros(K_new-1,dtype=int)
    
    for x in range(K_new-1):
        ances = ancestors[x]  # ancestors of the current node
        evaluated_set = set(f_ordering[0:x+1])  # features evaluated so far
        temp_Marginal = np.identity(Bins)
        for y in ances:
            if y in evaluated_set:
                Conditioned_on[x] = y
                break
            else:
                temp_Marginal = np.matmul(temp_Marginal, CPDs[y-1,:,:])
    
        Marginals[x,:] =  temp_Marginal
    return Marginals, Conditioned_on

def find_ancestors(f_ordering,graph):
    # find ancestors in the graph
    K_new = len(f_ordering)
    ancestors =  [[] for _ in range(K_new-1)]
    
    for x in range(K_new-1):
        child = f_ordering[x+1]
        ancestors[x].append(child)
        while True:
            if graph[child] ==-1:
                break
            ancestors[x].append(graph[child])
            child = graph[child] 
    return ancestors

def reachable_beliefs(Marginals,CPD_r,f_ordering,Conditioned_on,P,L,Bins,beta):
    # get reachable belief points 
    K_new = len(f_ordering)
    max_feat = max(f_ordering)    
    SS = np.random.randint(0,Bins,size=(int(beta),max_feat+1))
            
    B = np.zeros((K_new-1,int(beta),L))
    Cond_ind = np.zeros((K_new-1,int(beta)),dtype=int)
    
    for z in range(int(beta)):         
        pin = P        
        obs = np.array(SS[z])
        for i in range(K_new-1):         
            index = obs[f_ordering[i]]
            
            if i ==0: # if root node
                prob = CPD_r[:,index]                  
            else:                
                cond_paren = Conditioned_on[i-1]
                index_p = obs[cond_paren]               
                prob = Marginals[i-1,:,index,index_p]
                
            pin = np.divide(np.multiply(pin,prob),np.dot(pin,prob))  
            
            Cond_ind[i,z] = obs[Conditioned_on[i]]
            B[i,z,:] = pin
    return B,Cond_ind

def backup(b,Alpha_prev,Q,c,L,V,TP):   
    # backup operator
    alpha_s = Q[np.argmin(np.dot(Q,b))] # alpha vector for stop action 
    alpha_c =  c*np.ones(L)    
    
    for k in range(V):
        VV = np.multiply(Alpha_prev,TP[:,k])
        Vec_O = VV[np.argmin(np.dot(VV,b))]
        
        alpha_c = np.add(alpha_c,Vec_O) #alpha vector for continue action
    
    G = [alpha_s,alpha_c]   
    #Optimal action,alpha vector, value
    Action = np.argmin(np.dot(G,b))  
    Alpha = G[Action]
    Value = np.dot(Alpha,b)   
    return Action,list(Alpha),Value

def find_range(ll,Edg):
    # get bin index
    for i,f in enumerate(Edg):
        if ll < f:
            return i
    return len(Edg)-1
