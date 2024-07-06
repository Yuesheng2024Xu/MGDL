import multigrade_dnn_classification as m_dnn
import numpy as np
#import matplotlib.pyplot as plt
import time
from dataset import get_mnist
#from scipy import signal




def multigrade_dnn_model(nn_parameter, opt_parameter, orgdata):
        
    """
    implement a multigrade linear composition model 
    
    Parameters
    ----------
    orgidata :          dictionary 
                        the information of orginal data  (train_X, train_Y, test_X, test_Y)          
    nn_parameter :      dictionary
                        the information of model (structure of network, regularization parameters)
    opt_parameter :     dictionary
                        the information of optimization 
                        containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)
        
    Returns
    -------
    trained_variable :  dictionary
                        updated pretrained for fixed information 
        
    """                                 
    grade_length = len(nn_parameter['mul_layers_dims'])           # the length of new grade will be trained
    
    
    
    trained_variable = {}

    
    #trained the new layer
    for grade in range(1,  grade_length + 1):
        
        
        print("\n----------------------grade : {}---------------------\n".format(grade))

        layers_dims = nn_parameter["mul_layers_dims"][grade-1]
        max_learning_rate = opt_parameter["max_learning_rate"]
        min_learning_rate = opt_parameter["min_learning_rate"]
        epochs = opt_parameter["epochs"][grade-1]
        
        if grade==1:
            data = orgdata
            history_dic, prepare_next_grade = m_dnn.multigrade_dnn_model_1(data, layers_dims, opt_parameter, max_learning_rate, min_learning_rate, epochs, grade)
        else:
            data = prepare_next_grade
            history_dic, prepare_next_grade = m_dnn.multigrade_dnn_model_ell(orgdata, data, layers_dims, opt_parameter, max_learning_rate, min_learning_rate, epochs, grade, train_previous, validation_prevoius)
            
        
        train_previous =  history_dic["train_predict"] 
        validation_prevoius = history_dic["validation_predict"]        
        
        trained_variable["grade"+str(grade)] = history_dic


    
    return trained_variable


      
