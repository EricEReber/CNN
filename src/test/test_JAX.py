import jax.numpy as jnp 
from jax import grad, jit 
from jax import random 
import time
from activationFunctions import *

def sigmoid_jax(x): 
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

def CostLogReg(target): 

    def func(X):
        return -(1.0 / target.shape[0] * jnp.sum(
            (target * jnp.log(X + 10e-10)) + (1 - target * jnp.log(1 - X + 10e-10))))

    return func


def CostOLS(target): 

    def func(X): 
        return (1.0 / target.shape[0]) * jnp.sum((target - X) ** 2)

    return func


def CostCrossEntropy(target): 

    def func(X): 
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func


if __name__ == "__main__": 
  

    targets = jnp.array([1, 0 ,1, 1, 0, 0, 0, 1])
    print(targets.reshape((-1,1)))
    
    cost_derv = grad(CostLogReg(targets))
    cost_mse = grad(CostOLS(targets)) 
    cost_CSE = grad(CostCrossEntropy(targets)) 

    inputs = jnp.array(np.random.rand(8,4))
    print(inputs)
    
    print('Derivative of CostLogReg using Jax')
    print(cost_derv(inputs.reshape((-1,1))))
   
    print('Derivative of MSE using Jax')
    print(cost_mse(inputs.reshape((-1,1))))   

    print('Derivative of CostCrossEntropy using Jax') 
    print(cost_CSE(inputs.reshape((-1,1))))
