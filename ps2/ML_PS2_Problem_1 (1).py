#!/usr/bin/env python
# coding: utf-8

# ## Problem 1: SPAM, SPAM, HAM 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt


# In[2]:


spam_train = np.loadtxt('spam_train.data', delimiter=',')
spam_validation = np.loadtxt('spam_validation.data', delimiter=',')
spam_test = np.loadtxt('spam_test.data', delimiter=',')


# In[3]:


def preprocess(data):
    m, n = data.shape
    
    # X,y split
    X = data[:, :n-1]
    y = data[:, n-1:]
    
    # Set y = -1
    y = np.apply_along_axis(lambda x: -1 if x == 0 else 1, 1, y).reshape(-1, 1)
    
    return X,y


# In[4]:


X_train, y_train = preprocess(spam_train)
X_validate, y_validate = preprocess(spam_validation)
X_test, y_test = preprocess(spam_test)


# In[5]:


def get_accuracy(X, y, w, b, λ_ =None, y_ = None, X_= None, σ2=None):
    def gaussian_kernel(x, y, σ2):
        return np.exp(-np.linalg.norm(x-y)**2 / (2 * σ2))
    m, n = X.shape
    if w is not None:
        z = np.dot(X,w) + b
        f = (y * z) > 0
    else:
        print(f"Computing accuracy for σ2 = {σ2}")
        y_predict = np.zeros(m)
        for i in range(m):
            wx = 0
            for λ, sl, sv in zip(λ_, y_, X_):
                wx += λ * sl * gaussian_kernel(X[i], sv, σ2)
            y_predict[i] = wx
        y_predict + b
        f = (y_predict * y.ravel()) > 0
    return np.sum(f.astype('float32')) * 100/m


# In[6]:


# Primal problem
def SVM_primal(X, Y, c):
    print(f"Computing c = {c}")
    m, n = X.shape
    
    P = np.zeros((m+n+1, m+n+1))
    P[:n,:n] = np.eye(n,n)
    P = cvxopt.matrix(P)
    
    q = np.zeros((m+n+1, 1))
    q[n:m+n,0] = c
    q = cvxopt.matrix(q)
    
    G = np.zeros((2*m, m+n+1))
    for i in range(m):
        for j in range(n):
            G[i][j] = -1 * Y[i] * X[i][j]
        G[i][n+i] = -1
        G[i][m+n] = -1 * Y[i]
        G[m+i][n+i] = -1
    G = cvxopt.matrix(G)
    
    h = np.zeros((2*m, 1))
    h[:m,0] = -1
    h = cvxopt.matrix(h)
    
    # CVXOPT Solver
    solution = cvxopt.solvers.qp(P, q, G, h)
    sol = np.array(solution['x'])
    w = sol[:n]
    b = sol[m+n]
    return w, b
   
    
# Dual problem
def SVM_dual(X, Y, c, σ2):

    print(f"Computing c = {c} and variance = {σ2}")
    m, n = X.shape
    
    # Create Gram matrix
    K = np.zeros((m, m))
    # Use gaussian kernel
    X_sq = -2 * np.dot(X, X.T)
    X_sq += (X ** 2).sum(axis=1).reshape(-1, 1)
    X_sq += (X ** 2).sum(axis=1)

    K = X_sq / (-2 * σ2)
    np.exp(K, K)
    # P = Combination of Yi Yj Xi Xj
    P = cvxopt.matrix(np.outer(Y,Y) * K)
    q = cvxopt.matrix(np.ones(m) * -1)

    # Constraints λY = 0
    A = cvxopt.matrix(Y, (1,m), 'd')
    b = cvxopt.matrix(0.0)
    # λ >= 0
    lhs = np.diag(np.ones(m) * -1)
    lhs2 = np.identity(m)
    G = cvxopt.matrix(np.vstack((lhs, lhs2)))
    rhs = np.zeros(m)
    rhs2 = np.ones(m) * c
    h = cvxopt.matrix(np.hstack((rhs, rhs2)))

    # CVXOPT Solver
    solution = cvxopt.solvers.qp(P, q, G, h, A, b, options={'show_progress':False})

    # Solver produces λ
    λ = np.ravel(solution['x'])
    n_λ = len(λ)
    sv = λ > 1e-5
    idx = np.arange(n_λ)[sv]
    l = λ[sv]
    support_labels = Y[sv]
    support_vectors = X[sv]
    
    b = 0.0
    for n in range(len(l)):
        b += support_labels[n]
        b -= np.sum(l * support_labels * K[idx[n],sv])
    b /= len(l)
        
    return b, l, support_labels, support_vectors


# In[9]:


def run_svm(c_list, σ2_list = [None], primal = True):
    data = {
        'c': [],
        'variance': [],
        'Training Data Accuracy': [],
        'Validation Data Accuracy': []
    }
    best_c, best_σ2, best_validation_acc, best_train_acc = c_list[0], σ2_list[0], 0, 0
    best_w, best_b = [], 0
    for c in c_list:
        for σ2 in σ2_list:
            data['c'].append(c)
            data['variance'].append(σ2)
            w, b, λ, support_labels, sv = None, None, None, None, None
            if primal:
                w, b = SVM_primal(X_train, y_train, c)
                train_acc = get_accuracy(X_train, y_train, w, b)
                validation_acc = get_accuracy(X_validate, y_validate, w, b)
            else:
                w = None
                b, λ, support_labels, sv = SVM_dual(X_train, y_train ,c,σ2)
                train_acc = get_accuracy(X_train, y_train, None, b, λ_ = λ, y_ = support_labels, X_ = sv, σ2 = σ2)
                validation_acc = get_accuracy(X_validate, y_validate, None, b, λ_ = λ, y_ = support_labels, X_ = sv, σ2 = σ2)
            data['Training Data Accuracy'].append(train_acc)
            data['Validation Data Accuracy'].append(validation_acc)
            if validation_acc > best_validation_acc or (validation_acc == best_validation_acc and train_acc > best_train_acc):
                best_validation_acc = validation_acc
                best_train_acc = train_acc
                best_c = c
                best_σ2 = σ2
                best_w = w
                best_b = b
                best_λ = λ
                best_sv = sv
                best_sv_y = support_labels
    df = pd.DataFrame.from_dict(data)
    if primal:
        test_acc = get_accuracy(X_test, y_test, best_w, best_b)
    else:
        test_acc = get_accuracy(X_train, y_train, None, best_b, λ_ = best_λ, y_ = best_sv_y, X_ = sv, σ2 = σ2)
    df['Testing Data Accuracy'] = df.apply(lambda row: test_acc if row['c'] == best_c and row['variance'] == best_σ2 else None, axis=1)
    if best_σ2 is None:
        df.drop(columns=['variance'], inplace=True)
    return df


# **1. Primal SVMs**
# 
# * Using gradient descent or quadratic programming, apply the SVM with slack formulation to train a classiﬁer for each choice of $c ∈{1,10,10^{2},10^{3},10^{4},10^{5},10^{6},10^{7},10^{8}}$ without using any feature maps. 
# 
# * What is the accuracy of the learned classiﬁer on the training set for each value of c? 
# 
# * Use the validation set to select the best value of c. What is the accuracy on the validation set for each value of c? 
# 
# * Report the accuracy on the test set for the selected classiﬁer.

# In[10]:


c_list = [1,10,10**2,10**3,10**4,10**5,10**6,10**7,10**8]
#df_primal = run_svm(c_list, primal=True)
#df_primal.to_csv('primal.csv', index=False)


# In[11]:


#df_primal.style.apply(lambda x: ['background: lightgreen' if not np.isnan(x['Testing Data Accuracy']) else '' for i in x], axis=1)


# **2. Dual SVMs with Gaussian Kernels**
# 
# * Using quadratic programming, apply the dual of the SVM with slack formulation to train a classiﬁer for each choice of c $c ∈{1,10,10^{2},10^{3},10^{4},10^{5},10^{6},10^{7},10^{8}}$ using a Gaussian kernel with $σ^{2} ∈{.1,1,10,100,1000}$. 
# * What is the accuracy of the learned classiﬁer on the training set for each pair of c and σ? 
# * Use the validation set to select the best value of c and σ. What is the accuracy on the validation set for each pair of c and σ? 
# * Report the accuracy on the test set for the selected classiﬁer.
# 

# In[ ]:


c_list = [1,10,10**2,10**3,10**4,10**5,10**6,10**7,10**8]
σ2_list = [.1,1,10,100,1000]
df_dual = run_svm(c_list, σ2_list, primal=False)


# In[ ]:


df_dual.style.apply(lambda x: ['background: lightgreen' if not np.isnan(x['Testing Data Accuracy']) else '' for i in x], axis=1)
df_dual.to_csv('dual.csv', index=False)


# **3. k-Nearest Neighbors**
# 
# * What is the accuracy of the k-nearest neighbor classiﬁer for k = 1,5,11,15,21? 

# In[ ]:


from collections import defaultdict

class kNN:
    
    def __init__(self, neighbors=1):
        self.train_data = []
        self.labels = []
        self.m = 0
        self.n = 0
        self.k = neighbors
    
    def fit(self, X_train, y_train):
        self.train_mean = X_train.mean(axis=0)
        self.train_std = X_train.std(axis=0)
        self.train_data = X_train
        self.train_data_normed = (self.train_data - self.train_mean) / self.train_std 
        self.m, self.n = self.train_data.shape
        self.labels = y_train
    
    def get_distances(self, X_test):
        distances = -2 * self.train_data_normed.dot(X_test.T) + np.sum(X_test**2,axis=1) + np.sum(self.train_data_normed**2,axis=1)[:, np.newaxis]
        distances[distances < 0] = 0
        return distances
    
    def predict(self, X_test):
        X_test_normed = (X_test - self.train_mean) / self.train_std
        distances = self.get_distances(X_test_normed)
        idx = np.argsort(distances, axis=0)
        idx = idx[0:self.k, :]
        m, n = idx.shape
        labels = self.labels.ravel()
        y_pred = np.zeros((X_test.shape[0], 1))
        for col in range(n):
            classes = defaultdict(int)
            for row in range(m):
                label = labels[idx[row, col]]
                classes[label] += 1
            # Get the majority class
            y_pred[col] = max(classes, key=classes.get)
        return y_pred
    
    @staticmethod
    def get_accuracy(y_pred, y_test):
        return np.mean(y_pred.flatten() == y_test.flatten()) * 100
        


# In[ ]:


def run_kNN(k_list):
    data = {
        'k': [],
        'Validation Accuracy': [],
        'Test data Accuracy': []
    }
    for k in k_list:
        data['k'].append(k)
        classifier = kNN(k)
        classifier.fit(X_train, y_train)
        y_validate_pred = classifier.predict(X_validate)
        data['Validation Accuracy'].append(kNN.get_accuracy(y_validate_pred, y_validate))
        y_test_pred = classifier.predict(X_test)
        data['Test data Accuracy'].append(kNN.get_accuracy(y_test_pred, y_test))
        
    return pd.DataFrame.from_dict(data)


# In[ ]:


k_list = [1,5,11,15,21]
df = run_kNN(k_list)
display(df)


# **4. Which of these approaches (if any) should be preferred for this classiﬁcation task? Explain**

# SVM with gaussian kernel should be prefered for this classification task as it performs better in Higher dimensions compared to kNN.
