import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple


@dataclass
class Linear:
  in_size: int
  out_size: int

  is_mat: bool = True
  weights: np.ndarray = np.zeros(1)
  bias: np.ndarray = np.zeros(1)
  
  weights_dx: np.ndarray = np.zeros(1)
  bias_dx: np.ndarray = np.zeros(1)

  active_dx: np.ndarray = np.zeros(1)

  def __post_init__(self) -> None:
     self.weights = np.random.randn(self.in_size, self.out_size).T
     self.bias = np.zeros((1, self.out_size)).T

  def __call__(self, X, update=True) -> np.ndarray:
    '''
    Apply Linear function on vector, i.e Wx + b = z

    Derivatives are:
    dz/dW = x^T
    dz/db = 1

    dz/dh = W^T

    params
    - update: updates derivative parameters with respect to weight and bias
    '''
    if update: 
      self.weights_dx = X.T
      self.bias_dx = 1
      self.active_dx = self.weights.T

    return self.weights @ X + self.bias

@dataclass
class Sigmoid:
  active_dx: np.ndarray = np.zeros(1)

  is_mat: bool = False

  def __call__(self, X, update=True) -> np.ndarray:
    '''
    Applies Sigmoid on numpy array, sig(z) = h

    Derivatives are:
    dh/dz = h(1-h)
    '''
    sigX = 1/(1 + np.exp(-X))

    if update: 
      self.active_dx = sigX * (1-sigX)

    return sigX

@dataclass
class Tanh:
  active_dx: np.ndarray = np.zeros(1)

  is_mat: bool = False

  def __call__(self, X, update=True) -> np.ndarray:
    '''
    Applies Tanh on numpy array, tanh(z) = h

    Derivatives are:
    dh/dz = 1-h^2
    '''
    tanhX = np.tanh(X)

    if update: self.active_dx = 1 - tanhX**2

    return tanhX

@dataclass
class ReLU:
  active_dx: np.ndarray = np.zeros(1)

  is_mat: bool = False

  def __call__(self, X, update=True) -> np.ndarray:
    '''
    Applies ReLU on numpy array, relu(z) = h

    Derivatives are:
    dh/dz = {
      h <= 0: 0
      h  > 0: 1
    }
    '''
    reluX = np.maximum(0, X)

    if update: 
      self.active_dx = reluX.copy()
      self.active_dx[self.active_dx <= 0] = 0
      self.active_dx[self.active_dx != 0] = 1

    return reluX

@dataclass
class SoftMax:
  active_dx: np.ndarray = np.zeros(1)

  is_mat: bool = False

  def __call__(self, X, update=True) -> np.ndarray:
    '''
    Applies SoftMax on numpy array, sftmax(z) = h

    Derivatives are:
    dh/dz = s_i(1-sj)
    '''
    softX = np.exp(X)/np.sum(np.exp(X))

    if update: self.active_dx = softX*(1-softX)

    return softX


@dataclass
class CrossEntropyLoss:
  def __call__(self, y, ypred):
    '''
    Returns CE Loss and it's derivative on two numpy vectors,  sum(-y*log(ypred) - (1-y)*log(1-ypred))

    Derivatives are:
    dL/dypred = -(y-1)/(1-ypred) - (y/ypred)
    '''
    ce = np.sum(-y*np.log(ypred)-(1-y)*np.log(1-ypred))
    ce_dx = -(y-1)/(1-ypred) - (y/ypred)

    return ce, ce_dx

@dataclass
class MeanSquaredLoss:
  def __call__(self, y, ypred):
    '''
    Returns MSE Loss and it's derivative on two numpy vectors,  1/n sum((y-ypred)^2)

    Derivatives are:
    dL/dypred = (-2/n)*(y-ypred)
    '''
    mse = 1/np.size(y)*np.sum((y-ypred)**2)
    mse_dx = (-2/np.size(y)) * (y-ypred)

    return mse, mse_dx

@dataclass
class OneHotEncoding:
  def __call__(self, target):
    '''
    Applies one-hot encoding on categorical data
    '''
    encoding = np.zeros((target.size, np.max(target)+1))
    encoding[np.arange(target.size), target] = 1
    return encoding



class NeuralNet:
  def __init__(self) -> None:
    self.sequence_ = []
    self.hyperparameters_ = 0

  def Sequential(self, *fns) -> None:
    #Initializes sequential list
    self.sequence_ = fns
    self.hyperparameters = sum([np.size(block.weights)+np.size(block.bias) for block in self.sequence_ if block.is_mat])


  def forward(self, X, update_params=True) -> np.ndarray:
    for fn in self.sequence_:
      X = fn(X, update_params) 

    return X

  def backprop(self, loss_dx, lr) -> None:
    chain_stack = loss_dx
    for indx, block in enumerate(reversed(self.sequence_)):

      if block.is_mat:
        block.weights = block.weights - lr*(chain_stack @ block.weights_dx)
        block.bias = block.bias - lr*(chain_stack) #@ block.bias_dx) NOTE: linear derivative for bias is 1, other network structures may be different

        chain_stack = block.active_dx@chain_stack
      else:
        chain_stack = block.active_dx*chain_stack

  
  def train(self, X_train, y_train, lossfn, epochs=10, lr=0.001) -> dict:
    losses, accs = [], []
    for epoch in range(1, epochs+1):
      loss_train = 0
      acc = 0
      for data, labels in zip(X_train, y_train):
        outputs = self.forward(np.array([data]).T)
        loss, loss_dx = lossfn(np.array([labels]).T, outputs)
        
        self.backprop(loss_dx, lr)

        acc += np.argmax(outputs) == np.argmax(labels)

        loss_train += loss
      
      if epoch%(epochs//10)==0:
        acc = np.round(acc/len(X_train), 3)
        loss_train = np.round(loss_train, 3)
        print(f"Epoch #{epoch}\tLoss: {loss_train}\tAcc: {acc}")
        accs += [acc]
      losses += [loss_train]
      
    return {"loss_hist": losses, "acc_hist": accs}
  
  def __call__(self, X) -> np.ndarray:
    return self.forward(np.array([X]).T)
