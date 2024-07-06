# -*- coding: utf-8 -*-
"""
Created on Wed 12/31 2023

@author: rfang002
"""
# import sys
# sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
# import seaborn as sns
# sns.set()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from singlegrade_dnn_regression import singlegrade_model_forward, rse 
from data_generate import example1, example2, example3


def results_analysis(fullfilename, example):

    with open(fullfilename, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f)
        
        
    opt = history['opt']
    print(opt)
    #example = history['example']
    
    if example == 'example1':
        data = example1(opt)
    elif example == 'example2':
        data = example2(opt)
    elif example == 'example3':
        data = example3(opt)
    
    train_predict, _ = singlegrade_model_forward(data["train_X"], nn_parameter['layers_dims'], history['parameters'], nn_parameter["activation"] , nn_parameter["sinORrelu"])
    test_predict, _ = singlegrade_model_forward(data["test_X"], nn_parameter['layers_dims'], history['parameters'], nn_parameter["activation"] , nn_parameter["sinORrelu"])

    
    print("###########################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)

    print('train_rse is {}, validation_rses is {}, test res is {}'.format(history['train_rses'][-1], history['validation_rses'][-1], rse(data["test_Y"], test_predict)))

    print('the train time is {}'.format(history["time"])) 
    
    
    

    plt.plot(history["REC_FRQ_iter"], np.array(history["train_costs"]), label="vrain loss")
    plt.plot(history["REC_FRQ_iter"], np.array(history["validation_costs"]), label="validation loss")
    plt.xlabel('Number of training epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('SGDL: loss')
    # fig_filename = 'Fig/SGDL_Example_1_noisefree_loss.png'
    # plt.savefig(fig_filename)
    plt.show()
    
    plt.scatter(data['train_X'].T, data['train_Y'].T, s=5,label = 'true labels')
    plt.scatter(data['train_X'].T, train_predict.T, s=5, label = 'predict labels')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('SGDL')
    plt.show()


    plt.scatter(data['test_X'].T, data['test_Y'].T, s=5, label = 'true labels')
    plt.scatter(data['test_X'].T, test_predict.T, s=5, label = 'predict labels')
    plt.legend()
    plt.title('SGDL') 
    # fig_filename = 'Fig/SGDL_Example_1_noisefree_predict.png'
    # plt.savefig(fig_filename)
    plt.show()    

        
    return


