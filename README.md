# NeuralNet

Linear Neural Network library for Python created with pure NumPy

## Table of Contents
* [Features](#features)
* [Examples](#examples)
* [Component Template](#component-template)
* [Setup](#setup)
* [Requirements](#requirements)


## Features
- Lightweight library for general-purpose machine learning
- Easy-to-use sequential modelling, with 4+ components to choose from
- Components for holding backpropagation derivatives and/or weights+bias
- Simple implementation for easy development
- Includes One-hot encoding, MLE, CELoss and train function

## Examples
- [Iris Dataset](../main/examples/Iris.ipynb)
- [MNIST Dataset](../main/examples/MNIST.ipynb)

## Component Template

Standard templates for creating new components

### Activation function

```
@dataclass
class Activation:
  active_dx: np.ndarray = np.zeros(1)

  is_mat: bool = False

  def __call__(self, X, update=True) -> np.ndarray:
    '''
    Applies Activation on numpy array, activation(z) = h
    Derivatives are:
    dh/dz = ACTIVATION DERIVAITVE
    '''
    
    #ACTIVATION CODE
    activationX = ...

    if update: 
      #ACTIVATION DERIVATIVE CODE
      self.active_dx = ...

    return activationX
```

### Loss function

```
@dataclass
class CoolLoss:
  def __call__(self, y, ypred):
    '''
    Returns Cool Loss and it's derivative on two numpy vectors, MATH FOR LOSS FUNCTION
    Derivatives are:
    dL/dypred = LOSS DERIVATIVE
    '''
    #LOSS CODE
    cool = ...
    
    #LOSS DERIVATIVE CODE
    cool_dx = ...

    return cool, cool_Dx
```


## Setup

Download LinearNet.py and place in current directory. For Colab, drag into ```Fi;es```.

## Requirements

Latest version of ```numpy``` and ```dataclasses```




