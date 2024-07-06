# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 23:06:10 2023

@author: rfang002
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os.path
import singlegrade_dnn_regression as dnn
from argparse import Namespace
from data_generate import example1, example2, example3



def single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, example, noise):

    #---------neural network parameter--
    nn_parameter = {}   
    nn_parameter["layers_dims"] = layers_dims
    nn_parameter["lambd_W"] = 0
    nn_parameter["sinORrelu"] = 3
    nn_parameter["activation"] = "relu"
    nn_parameter["init_method"] = "xavier"
    #-----------------------------------
    
    
    
    #-------optimization parameter--
    opt_parameter = {}
    opt_parameter["optimizer"] = "adam"
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    opt_parameter["max_learning_rate"] = max_learning_rate
    opt_parameter["min_learning_rate"] = min_learning_rate
    opt_parameter["epochs"] = epochs
    opt_parameter["REC_FRQ"] = 1
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch_size
    #-----------------------------

    opt = Namespace()
    opt.ntrain = 5000
    opt.ntest = 1000
    opt.noise = noise
    opt.example = example
    
    if example == 'example1':
        data = example1(opt)
    elif example == 'example2':
        data = example2(opt)
    elif example == 'example3':
        data = example3(opt)
        
        
    history = dnn.singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter)
    history['opt'] = opt

    save_path = 'results/{}/noise{}'.format(example, opt.noise) 
    filename = "SGDL_xavier_noise%s_epochs%d_minibatch%s_MAXlearningrate%.2e_MINlearningrate%.2e_validation%.4e_train%.4e.pickle"%(opt.noise, epochs, opt_parameter["mini_batch_size"],
                                                                                                                                   opt_parameter["max_learning_rate"],
                                                                                                                                   opt_parameter["min_learning_rate"],
                                                                                                                                   history['validation_rses'][-1],
                                                                                                                                   history['train_rses'][-1])
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([history, nn_parameter, opt_parameter],f)






    











