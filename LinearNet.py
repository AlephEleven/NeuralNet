import numpy as np
from dataclasses import dataclass, field
import datetime

'''
Layer Component
'''

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

'''
Activation Functions
'''

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
      self.active_dx[self.active_dx > 0] = 1

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
    softX = np.exp(X)/np.sum(np.exp(X), axis=0)

    if update: self.active_dx = softX*(1-softX)

    return softX


'''
Loss Functions
'''

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

'''
Gradient Descent
'''

@dataclass
class StochasticGradientDescent:
  lr: float = 0.01

  def __call__(self, learnable_block, chain_stack, *args):
    '''
    Uses SGD to update the weights/bias of a given block
    
    Algorithm is:
    w(t+1) = w(t) - lr*dL(w)
    '''
    dw = chain_stack @ learnable_block.weights_dx
    learnable_block.weights -= self.lr*dw

    db = np.mean(chain_stack, axis=1)[np.newaxis].T
    learnable_block.bias -= self.lr*db

@dataclass
class SGDMomentum:
  lr: float = 0.001
  momentum: float = 0.9

  def __call__(self, learnable_block, chain_stack, *args):
    '''
    Uses Momentum+SGD to update the weights/bias of a given block
    
    Algorithm is:
    step(t+1) = mu*step(t) + dL(w)
    w(t+1) = w(t) - lr*b(t+1)
    '''
    if 'step_weights' not in dir(learnable_block):
      learnable_block.step_weights = np.zeros_like(learnable_block.weights)

    dw = chain_stack @ learnable_block.weights_dx
    learnable_block.step_weights = self.momentum*learnable_block.step_weights + dw
    learnable_block.weights -= self.lr*learnable_block.step_weights
    
    if 'step_bias' not in dir(learnable_block):
      learnable_block.step_bias = np.zeros_like(learnable_block.bias)

    db = np.mean(chain_stack, axis=1)[np.newaxis].T
    learnable_block.step_bias = self.momentum*learnable_block.step_bias + db
    learnable_block.bias -= self.lr*learnable_block.step_bias

@dataclass
class RMSProp:
  alpha: float = 0.99
  lr: float = 0.01
  eps: float = 1e-8

  def __call__(self, learnable_block, chain_stack, *args):
    '''
    Uses RMSProp to update the weights/bias of a given block
    
    Algorithm is:
    vel(t+1) = alpha * vel(t) + (1-alpha)*dL(w)^2
    w(t+1) = w(t) - lr*dw/(sqrt( vel(t+1)+eps ))
    '''
    if 'velocity_weights' not in dir(learnable_block):
      learnable_block.velocity_weights = np.zeros_like(learnable_block.weights)

    dw = chain_stack @ learnable_block.weights_dx
    learnable_block.velocity_weights = self.alpha * learnable_block.velocity_weights + (1 - self.alpha) * dw**2
    learnable_block.weights -= self.lr * dw/(np.sqrt(learnable_block.velocity_weights) + self.eps)
    

    if 'velocity_bias' not in dir(learnable_block):
      learnable_block.velocity_bias = np.zeros_like(learnable_block.bias)

    db = np.mean(chain_stack, axis=1)[np.newaxis].T
    learnable_block.velocity_bias = self.alpha * learnable_block.velocity_bias + (1 - self.alpha) * db**2
    learnable_block.weights -= self.lr * db/(np.sqrt(learnable_block.velocity_bias) + self.eps)

@dataclass
class AdamOptimizer:
  lr: float = 0.001
  beta1: float = 0.9
  beta2: float = 0.999
  eps: float = 1e-8

  def __call__(self, learnable_block, chain_stack, epoch):
    '''
    Uses Adam Optimizer to update the weights/bias of a given block
    
    Algorithm is:
    momentum(t+1) = beta1 * momentum(t) + (1-beta1) * dL(w)
    velocity(t+1) = beta2 * velocity(t) + (1-beta2) * dL(w)^2

    momentum_corrected = momentum(t+1)/(1 - beta1^(current epoch))
    velocity_corrected = velocity(t+1)/(1 - beta2^(current epoch))

    w(t+1) = w(t) - lr*momentum_corrected/(sqrt( velocity_corrected+eps ))
    '''

    if 'momentum_weights' not in dir(learnable_block):
      learnable_block.momentum_weights = np.zeros_like(learnable_block.weights)

    if 'velocity_weights' not in dir(learnable_block):
      learnable_block.velocity_weights = np.zeros_like(learnable_block.weights)

    dw = chain_stack @ learnable_block.weights_dx
    learnable_block.momentum_weights = self.beta1 * learnable_block.momentum_weights + (1 - self.beta1) * dw
    learnable_block.velocity_weights = self.beta2 * learnable_block.velocity_weights + (1 - self.beta2) * dw**2

    momentum_weights_corrected = learnable_block.momentum_weights/(1 - self.beta1**(epoch+1))
    velocity_weights_corrected = learnable_block.velocity_weights/(1 - self.beta1**(epoch+1))
    learnable_block.weights -= self.lr * momentum_weights_corrected/(np.sqrt(velocity_weights_corrected) + self.eps)

    if 'momentum_bias' not in dir(learnable_block):
      learnable_block.momentum_bias = np.zeros_like(learnable_block.bias)

    if 'velocity_bias' not in dir(learnable_block):
      learnable_block.velocity_bias = np.zeros_like(learnable_block.bias)

    db = np.mean(chain_stack, axis=1)[np.newaxis].T
    learnable_block.momentum_bias = self.beta1 * learnable_block.momentum_bias + (1 - self.beta1) * db
    learnable_block.velocity_bias = self.beta2 * learnable_block.velocity_bias + (1 - self.beta2) * db**2

    momentum_bias_corrected = learnable_block.momentum_bias/(1 - self.beta1**(epoch+1))
    velocity_bias_corrected = learnable_block.velocity_bias/(1 - self.beta1**(epoch+1))
    learnable_block.bias -= self.lr * momentum_bias_corrected/(np.sqrt(velocity_bias_corrected) + self.eps)

'''
Neural Net Framework
'''

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

  def backprop(self, loss_dx, grad_descent, epoch) -> None:
    chain_stack = loss_dx
    for indx, block in enumerate(reversed(self.sequence_)):

      if block.is_mat:
        grad_descent(block, chain_stack, epoch)

        chain_stack = block.active_dx@chain_stack
      else:
        chain_stack = block.active_dx*chain_stack

  
  def train(self, training_data, lossfn, grad_descent, epochs=10, display=True, timed=True) -> dict:
    losses, accs = [], []
    for epoch in range(1, epochs+1):
      loss_train = 0
      acc = 0
      training_data.reshuffle()

      for data, labels in training_data:
        outputs = self.forward(data)
        loss, loss_dx = lossfn(labels, outputs)
        
        self.backprop(loss_dx, grad_descent, epoch)

        acc += np.sum(outputs.argmax(axis=0) == labels.argmax(axis=0))

        loss_train += loss
      
      if epoch%(epochs//10)==0:
        acc = np.round(acc/training_data.maxsize, 3)
        loss_train = np.round(loss_train, 3)
        current_time = datetime.datetime.now() if timed else ""
        if display: print(f"Epoch #{epoch}\tLoss: {loss_train}\tAcc: {acc}\t {current_time}")
        accs += [acc]
      losses += [loss_train]
      
    return {"loss_hist": losses, "acc_hist": accs}
  
  def __call__(self, X) -> np.ndarray:
    return self.forward(X[np.newaxis].T, update_params=False)

'''
Utility Code
'''

@dataclass
class OneHotEncoding:
  def __call__(self, target) -> np.ndarray:
    '''
    Applies one-hot encoding on categorical data
    '''
    encoding = np.zeros((target.size, np.max(target)+1))
    encoding[np.arange(target.size), target] = 1
    return encoding 

class DataLoader:
  '''
  Creates a generator from a set of features/labels and splits & shuffles data into batches

  params
  - batch_size: size of each batch
  - shuffle: check whether to shuffle data before loading
  '''
  def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 16, shuffle: bool = True) -> None:
    self.X = X
    self.y = y

    self.maxsize = len(X)
    self.batch_size = batch_size
    self.shuffle = shuffle
    
    self.batch_count = self.maxsize//self.batch_size if self.maxsize%self.batch_size==0 else self.maxsize//self.batch_size+1
    
    self.indices = np.arange(self.maxsize)

    self.reshuffle()

  def reshuffle(self):
    if self.shuffle: np.random.shuffle(self.indices)

    X_rand = self.X[self.indices]
    y_rand = self.y[self.indices]

    self.data = ([X_rand[i*self.batch_size:(i+1)*self.batch_size if (i+1)*self.batch_size < self.maxsize else self.maxsize].T,
                 y_rand[i*self.batch_size:(i+1)*self.batch_size if (i+1)*self.batch_size < self.maxsize else self.maxsize].T] for i in range(self.batch_count))

  def __iter__(self):
    return self.data

  def __next__(self) -> tuple:
    return next(self.data)