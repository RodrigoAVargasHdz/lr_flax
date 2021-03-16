import os
import argparse
import time
import datetime
import itertools

import numpy as onp
import numpy.random as onpr

import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from jax import value_and_grad, grad, jacfwd, jacrev
from jax.experimental import optimizers

from jax.config import config
config.update("jax_debug_nans", True)
jax.config.update('jax_enable_x64', True)

from flax.core.frozen_dict import freeze, unfreeze,FrozenDict
from flax import serialization

# from flax_mlp import MLP
from flax_lr import LR

Ha2cm = 220000

r_dir = 'TempResults'

# --------------------------------   
def data(N):
    a, b = 1., -2.
    X = onp.random.rand(N) + onp.linspace(-10.,10.,num=N)
    Y = (a*X + b) + onp.random.rand(N)
    return X[:,onp.newaxis],Y

# --------------------------------
def main_opt(N):

    start_time = time.time()
    
    n_epochs = 500

    str_nn_arq = '_1'
    nn_arg = (1)

#     Data
#     N = 1000#600
    X,Y = data(N)
    N = X.shape[0]

# --------------------------------
    print('-----------------------------------')
    print('Starting time' )
    print('-----------------------------------')
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f_out)
    print('N = {}, n_atoms = {}, random = {}'.format(N,n_atoms,i0))
    print(str_nn_arq)
    print('Batch Size = {}, N Epoch = {}'.format(batch_size,n_epochs))
    print('-----------------------------------')

#     --------------------------------------    
#     initialize NN
    tuple_nn_arq = nn_arq
    model = LR(tuple_nn_arq)

    def get_init_NN_params(key):     
        x = jnp.ones((X.shape[1]))
        mlp_variables = model.init(key, x)
        return mlp_variables

    @jit
    def f_loss(params):
        y_pred = model.apply(params, X)    
        diff_y = y_pred - Y #y-ytr #Ha2cm*
        z = jnp.mean(jnp.square(diff_y))#jnp.sqrt()
        return z 

#     --------------------------------------    
#     Initilialize parameters   
    rng = random.PRNGKey(0)
    rng, subkey = jax.random.split(rng)
    
    nn_params = get_init_NN_params(subkey)
    print(nn_params)

#     --------------------------------------
#     Optimization       
    init_params = nn_params#(lambd_params,)
    
    opt_init, opt_update, get_params = optimizers.adam(step_size = 2E-3,b1=0.5, b2=0.5)#
    opt_state = opt_init(init_params)
    
    def update(i, opt_state):
        params = get_params(opt_state)
        loss, g_params = value_and_grad(f_loss)(params)
#         print('grad params')
#         print(g_params)
        return loss, g_params, opt_update(i, g_params, opt_state) 
           
    loss0 = 1E16
    itercount = itertools.count()
    for epoch in range(n_epochs):
        loss, grad_params, opt_state = update(next(itercount), opt_state)
        params = get_params(opt_state)
        if loss < loss0:
            loss0 = loss
            nn_params = params

    print('---------------------------------' )
    print('Training time =  %.6f seconds ---'% ((time.time() - start_time)) )
    print('---------------------------------')

def main():
    N = 100	
	main_opt(N)    



if __name__ == "__main__":
    main()  
    
