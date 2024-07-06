
#import matplotlib.pyplot as plt
import time
import pickle
import os.path
import multigrade_dnn_model as m_dnn
from dataset import get_mnist, get_fashionmnist
from argparse import Namespace
import numpy as np


#------------------------------------nn_parameter--------------------------------

def multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, noise, structure, dataname):
                               
    #---------------------------neural network parameter----------------------------
    nn_parameter = {}
    nn_parameter["mul_layers_dims"] = mul_layers_dims         # neural network strucure

    #------------------------optimization parameter----------------------------------
    opt_parameter = {}
     #---------------default paramwter for Adam algoirthm----------------------------------
    opt_parameter["optimizer"] = "adam"                        # use Adam optimizer  
    opt_parameter["beta1"] = 0.9
    opt_parameter["beta2"] = 0.999
    opt_parameter["epsilon"] = 1e-8
    opt_parameter["error"] = 1e-7
    #-------------------------------------------------------------------------------------
    opt_parameter["SGD"] = SGD
    opt_parameter["mini_batch_size"] = mini_batch             # mini batch size
    opt_parameter["REC_FRQ"] = 1                              # record loss evry steps
    opt_parameter["init_method"] = "xavier"                   # use xavier initialization
    opt_parameter["max_learning_rate"] = max_learning_rate    # maximum learning rate
    opt_parameter["min_learning_rate"] = min_learning_rate    # minimum learning rate
    opt_parameter["epochs"] = mul_epochs                      # the training number of epoch in each grade
    #---------------------------------------------------------------------------------
    
    if dataname == "MNIST":
        data  = get_mnist(noise)
    elif dataname == "FashionMNIST":
        data = get_fashionmnist(noise)
        
        
    trained_variable = m_dnn.multigrade_dnn_model(nn_parameter, opt_parameter, data)


    save_path = 'results/{}noise{}/structure{}'.format(dataname, noise, structure)
 
    filename = "MGDL_%snoise%.1f_batchzie%s_epochs%s_MAXLrate%.2e_MINLrate%.2e_TrA%.4e_VaA%.4e.pickle"%(dataname, noise, opt_parameter["mini_batch_size"],opt_parameter['epochs'],
                                                                                                        opt_parameter["max_learning_rate"],opt_parameter["min_learning_rate"],
                                                                                                        trained_variable["grade"+str(len(mul_layers_dims))]["train_accuracy"],
                                                                                                        trained_variable["grade"+str(len(mul_layers_dims))]["validation_accuracy"])
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([trained_variable, nn_parameter, opt_parameter],f)





    












