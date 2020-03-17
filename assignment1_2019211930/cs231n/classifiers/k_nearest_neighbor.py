import numpy as np
from collections import Counter

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):

    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):

    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):#测试样本的循环
      for j in range(num_train):#训练样本的循环            
        #dists[i,j]=np.sqrt(np.sum(np.square(self.X_train[j,:]-X[i,:])))
        dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j,:] - X[i,:])))
        #np.square是针对每个元素的平方方法  
    return dists

  def compute_distances_one_loop(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #dists[i,:] = np.sqrt(np.sum(np.square(self.X_train-X[i,:]),axis = 1))
      dists[i,:]=np.linalg.norm(X[i,:]-self.X_train[:],axis=1)
    return dists

  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    """
    mul1 = np.multiply(np.dot(X,self.X_train.T),-2)   
    sq1 = np.sum(np.square(X),axis=1,keepdims = True)   
    sq2 = np.sum(np.square(self.X_train),axis=1)   
    dists = mul1+sq1+sq2    
    dists = np.sqrt(dists) 
    """
    dists += np.sum(np.multiply(X,X),axis=1,keepdims = True).reshape(num_test,1)
    dists += np.sum(np.multiply(self.X_train,self.X_train),axis=1,keepdims = True).reshape(1,num_train)
    dists += -2*np.dot(X,self.X_train.T)
    dists = np.sqrt(dists) 

    return dists

  def predict_labels(self, dists, k=1):

    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):

        closest_y = []
        closest_y = self.y_train[np.argsort(dists[i, :])[:k]].flatten()
        c = Counter(closest_y)
        y_pred[i]=c.most_common(1)[0][0]
        """
        closest_y=self.y_train[np.argsort(dists[i, :])[:k]]      
        y_pred[i] = np.argmax(np.bincount(closest_y))
        """

    return y_pred

