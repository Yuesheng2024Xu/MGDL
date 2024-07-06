import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import itertools
import sys
from multigrade_dnn_model import multigrade_dnn_model_predict
from data_generate import example1, example2, example3
import sys
# sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.7/site-packages')
# import seaborn as sns
# sns.set()
    

def results_analysis(fullfilename, Figure, example):

    with open(fullfilename, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)
        
    
    print("################################################################################################################")
    print(fullfilename)
    print(nn_parameter)
    print(opt_parameter)
    
    
    #example = trained_variable['example']
    opt = trained_variable['opt']
    
    if example == 'example1':
        data = example1(opt)
    elif example == 'example2':
        data = example2(opt)
    elif example == 'example3':
        data = example3(opt)
    
    test_rse, predict_test_Y = multigrade_dnn_model_predict(data, nn_parameter, opt_parameter, trained_variable)    
    

    num_iter = trained_variable["REC_FRQ_iter"]

    train_rse = []
    validation_rse = []
    TRAIN_loss = []
    VALIDATION_loss = []
    MUL_EPOCH = []

    total_time =  0

    for i in range(0, len(nn_parameter["mul_layers_dims"])):       
        total_time = total_time + trained_variable['train_time'][i]
        train_rse.append(trained_variable['train_rse'][i][-1])
        validation_rse.append(trained_variable['validation_rse'][i][-1])
        if i==0:
            current_epoch = opt_parameter["epochs"][i]
            MUL_EPOCH.append(current_epoch)
            TRAIN_loss.extend(trained_variable['train_costs'][i])
            VALIDATION_loss.extend(trained_variable['validation_costs'][i])
        else:
            current_epoch += opt_parameter["epochs"][i]
            MUL_EPOCH.append(current_epoch)
            TRAIN_loss.extend(trained_variable['train_costs'][i][1:])
            VALIDATION_loss.extend(trained_variable['validation_costs'][i][1:])

    print('the train rse for each grade is {}'.format(train_rse))
    print('the validation rse for each grade is {}'.format(validation_rse))
    print('the test rse for each grade is {}'.format(test_rse))
    print('the train times for each grade is {}'.format(trained_variable['train_time']))
    print('the total train times is {}'.format(total_time))
    
    # print(TRAIN_loss)
    # print(VALIDATION_loss)
    
    plt.plot(np.array(TRAIN_loss), label='train loss')
    plt.plot(np.array(VALIDATION_loss),  label='validation loss')
    plt.xlabel("Number of training epochs")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.xlim([0,  MUL_EPOCH[2]])
    plt.legend()
    plt.title('MGDL: Loss') 
    plt.axvline(x=MUL_EPOCH[0], color='k', linestyle=':')
    plt.axvline(x=MUL_EPOCH[1], color='k', linestyle=':')
    #plt.axvline(x=600, color='k', linestyle=':')
    plt.xticks([0, MUL_EPOCH[0], MUL_EPOCH[1], MUL_EPOCH[2]]) 
    

    plt.tight_layout()
    # fig_filename = 'Fig/MGDL_Example_1_noisefree_loss.png'
    # plt.savefig(fig_filename)
    plt.show()

    
    if Figure:


        for i in range(0, len(nn_parameter["mul_layers_dims"])):


            num_iter = trained_variable["REC_FRQ_iter"][i]

            
            plt.scatter(data['train_X'].T, data['train_Y'].T, s=5,label = 'true labels')
            plt.scatter(data['train_X'].T, trained_variable["train_predict"][i][0].T, s=5, label = 'predict labels')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('MGDL: grade {}'.format(i+1))

            plt.show()


            plt.scatter(data['test_X'].T, data['test_Y'].T, s=5, label = 'true labels')
            plt.scatter(data['test_X'].T, predict_test_Y[i][0].T, s=5, label = 'predict labels')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title('MGDL: grade {}'.format(i+1)) 
            # fig_filename = 'Fig/MGDL_Example_1_noisefree_grade{}.png'.format(i+1)
            # plt.savefig(fig_filename)
            plt.show()
            
            
            


    return 




        
    


    
    
