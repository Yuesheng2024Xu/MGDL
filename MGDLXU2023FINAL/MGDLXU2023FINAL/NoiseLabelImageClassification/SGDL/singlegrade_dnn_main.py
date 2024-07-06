
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os.path
import singlegrade_dnn_classification as dnn
from argparse import Namespace
from dataset import get_mnist, get_fashionmnist



def single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, noise, structure, dataname):

                     
    nn_parameter = {}   
    nn_parameter["layers_dims"] = layers_dims
    nn_parameter["activation"] = "relu"
    nn_parameter["init_method"] = "xavier"
    
    
    
    #------------------------optimization parameter----------------------------------
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
    
    
    if dataname == "MNIST":
        data  = get_mnist(noise)
    elif dataname == "FashionMNIST":
        data = get_fashionmnist(noise)
        
        
    history = dnn.singlegrade_dnn_model_grade(data, nn_parameter, opt_parameter)

    save_path = 'results/{}noise{}/structure{}'.format(dataname, noise, structure)
    filename = "SGDL_noise%.1f_batchzie%s_epochs%d_MAXLrate%.2e_MINLrate%.2e_TrA%.4e_TeA%.4e.pickle"%(noise, opt_parameter["mini_batch_size"], opt_parameter['epochs'], 
                                                                                                      opt_parameter["max_learning_rate"], opt_parameter["min_learning_rate"],
                                                                                                      history['train_acc'][-1],history['validation_acc'][-1])
    fullfilename = os.path.join(save_path, filename)    
    
    with open(fullfilename, 'wb') as f:
        pickle.dump([history, nn_parameter, opt_parameter],f)
        
        
    print(fullfilename)






    











