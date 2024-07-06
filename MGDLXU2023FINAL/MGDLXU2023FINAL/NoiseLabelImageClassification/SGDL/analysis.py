
import sys
import os.path
import shutil
import os
# sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
# import seaborn as sns
# sns.set()

from dataset import get_mnist

import pickle
import numpy as np
import matplotlib.pyplot as plt
# from singlegrade_dnn_regression import singlegrade_model_forward, rse 


def results_analysis(SGDLfile, structure, noise):

    with open(SGDLfile, 'rb') as f:
        history, nn_parameter, opt_parameter = pickle.load(f) 
        

    print(SGDLfile)  
    print(nn_parameter)
    print(opt_parameter)

    SGDL_epochs = opt_parameter['epochs']
    SGDL_time  = history['time'][-1]


    print('(tmax, tmin) = ({}, {})'.format(opt_parameter["max_learning_rate"], opt_parameter["min_learning_rate"]))


    print('train accuracy is {}, validation accuracy is {}'.format(history['train_acc'][-1], history['validation_acc'][-1]))
    print('SGDL training time is {}'.format(SGDL_time))

    validation_acc = history["validation_acc"]
    train_acc = history["train_acc"]


    plt.plot(validation_acc, label='test accuracy')
    plt.plot(train_acc,  label='train accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.ylim([0.6, 0.9])
    #plt.ylim([0.65, 1.01])
    plt.ylim([0.4, 1.01])
    #plt.ylim([0.05, 1.01])
    plt.legend()
    plt.title('SGDL: ({}, {})'.format(opt_parameter["max_learning_rate"], opt_parameter["min_learning_rate"]))
    plt.tight_layout()


    # Save the figure
    # max_lr_str = '{:.0e}'.format(opt_parameter["max_learning_rate"]).replace('e-0', 'e-')
    # min_lr_str = '{:.0e}'.format(opt_parameter["min_learning_rate"]).replace('e-0', 'e-')
    # fig_filename = 'Fig/MINST_noise{:.1f}_S{}_SGDL_{}{}.png'.format(noise, structure, max_lr_str, min_lr_str)
    # fig_filename = 'Fig/MINST_noise%.1f_S%d_SGDL_%.0e%.0e.png'%(noise, structure,str(opt_parameter["max_learning_rate"]).replace('.', '').replace('e-0', 'e-'),
    #                                                         str(opt_parameter["min_learning_rate"]).replace('.', '').replace('e-0', 'e-'))
    #plt.savefig(fig_filename)

    plt.show()


