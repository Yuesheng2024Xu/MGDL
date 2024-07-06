#import matplotlib.pyplot as plt
import time
import pickle
import os.path
import multigrade_dnn_model as m_dnn
from data_generate import example1, example2, example3
from argparse import Namespace
import numpy as np


#------------------------------------nn_parameter--------------------------------

def multi_grade_dnn(MAX_learning_rate, MIN_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, activation, example, noise):
                               
    #---------------------------neural network parameter----------------------------
    nn_parameter = {}
    nn_parameter["mul_layers_dims"] = mul_layers_dims               # neural network strucure
    nn_parameter["activation"] = activation
    
    #------------------------optimization parameter----------------------------------
    opt_parameter = {}
    #---------------default paramwter for Adam algoirthm----------------------------------
    opt_parameter["optimizer"] = "adam"                           # use Adam optimizer
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    #----------------------------------------------------
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch              # mini batch size
    opt_parameter["REC_FRQ"] = 1                               # record loss evry 100 steps
    opt_parameter["init_method"] = "xavier"                    # use xavier initialization
    opt_parameter["MAX_learning_rate"] = MAX_learning_rate     # maximum learning rate
    opt_parameter["MIN_learning_rate"] = MIN_learning_rate     # minimum learning rate
    opt_parameter["epochs"] = mul_epochs                       # the training number of epoch in each grade

    #---------------------------------------------------------------------------------
    
    #----------------------------------record train history---------------------------
    trained_variable = {}                                     # store the different true lable and output for each grade  
    trained_variable['example'] = example                     # store example information
    trained_variable["train_time"] = []                       # store train time for each grade
    trained_variable["mul_parameters"] = []                   # store parameter for each grade                  
    trained_variable["train_rse"] = []
    trained_variable["validation_rse"] = [] 
    trained_variable["train_mse"] = []
    trained_variable["validation_mse"] = [] 
    trained_variable["train_predict"] = []
    trained_variable["validation_predict"] = []
    trained_variable["train_costs"] = []
    trained_variable["validation_costs"] = []  
    trained_variable["REC_FRQ_iter"] = []
    
    
    
    opt = Namespace()
    opt.ntrain = 5000
    opt.ntest = 1000
    opt.noise = noise
    
    if example == 'example1':
        data = example1(opt)
    elif example == 'example2':
        data = example2(opt)
    elif example == 'example3':
        data = example3(opt)
        

    
    trained_variable = m_dnn.multigrade_dnn_model(data, nn_parameter, opt_parameter, trained_variable)
    trained_variable['opt'] = opt
    
    
    #save results
    save_path = 'results/{}/noise{}'.format(example, opt.noise) 
    filename = "MGDL_xavier_noise%s_epoch%s_minibatch%s_MAXlearningrate%.4e_MINlearningrate%.4e_validation%.4e_train%.4e.pickle"%(opt.noise, opt_parameter["epochs"], opt_parameter["mini_batch_size"],
                                                                                                                                  opt_parameter["MAX_learning_rate"][0],opt_parameter["MIN_learning_rate"][-1],
                                                                                                                                  trained_variable['validation_rse'][-1][-1],
                                                                                                                                  trained_variable['train_rse'][-1][-1])
    fullfilename = os.path.join(save_path, filename) 
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([trained_variable, nn_parameter, opt_parameter],f)





    












