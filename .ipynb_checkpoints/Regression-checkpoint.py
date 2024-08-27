import numpy as np
import math
import random
import pandas as pd



class Regression():
    def __init__(self,kernel="linear"):
        self.W = np.array([])
        self.B = 0
        self.kernel = kernel

    def prediction(self,X):
        if self.kernel == "square":
            X_new = np.c_[X, X**2]
        elif self.kernel == "cubic":
            X_new = np.c_[X, X**2, X**3]
        elif self.kernel == "linear":
            X_new = X

        
        if self.W.shape[0] != X_new.shape[1]:
            self.W = np.zeros((X_new.shape[1],))
        if X_new.shape[1] == 1:
            return np.dot(X_new,self.W) + self.B
        else:
            return np.matmul(X_new,self.W) + self.B

    def _prediction(self,X):
        if self.W.shape[0] != X.shape[1]:
            self.W = np.zeros((X.shape[1],))
        if X.shape[1] == 1:
            return np.dot(X,self.W) + self.B
        else:
            return np.matmul(X,self.W) + self.B

    def cost(self,X,y):
        m = self.X.shape[0]
        return (1/(2*m))*(((self.prediction(X)-y)**2).sum())

    
    def _cost(self,X,y):
        m = self.X.shape[0]
        return (1/(2*m))*(((self._prediction(X)-y)**2).sum())

    def _compute_gradient(self,X,y):
        m,n = X.shape
        dj_dw = np.zeros((n,))
        for i in range(0,n):
            dj_dw[i] = ((1/m)*(((self._prediction(X)-y)*X[:,i]).sum()))
        dj_db = (1/m)*(((self._prediction(X)-y)).sum())
        return dj_dw,dj_db

    def train(self,X,y,alpha,tol,n_iters):
        m,n = X.shape
        if self.kernel == "square":
            self.X = np.c_[X, X**2, X**3]
        elif self.kernel == "cubic":
            self.X = np.c_[X, X**2, X**3]
        elif self.kernel == "linear":
            self.X = X
        j=0
        while (self._cost(self.X,y)>tol and j<n_iters):
            dj_dw,dj_db = self._compute_gradient(self.X,y)
            for i in range(0,self.X.shape[1]):
                self.W[i] = self.W[i]- alpha*dj_dw[i]
            self.B = self.B - alpha*dj_db
            j +=1
            if j%10000.0 == 0:
                print(j,self._cost(self.X,y)) 
        print(j,self._cost(self.X,y))





def max_normalize(X):
    X_new = X.copy().astype("float")
    
    for i in range(0,X_new.shape[1]):
        X_new[:,i] = X_new[:,i]/X_new[:,i].max()
    return X_new





def mean_normalize(X):
    X_new = X.copy().astype("float")
    for i in range(0,X_new.shape[1]):
        X_new[:,i] = (X_new[:,i]-X_new[:,i].mean())/(X_new[:,i].max()-X_new[:,i].min())
    return X_new