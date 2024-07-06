
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
import shutil
import os
import itertools
# import sys
# sys.path.append('/home/rfang002/envs/default-tensorflow-cpu-2.6.0/lib/python3.10/site-packages')
# import seaborn as sns
# sns.set()



def results_analysis(MGDLfile, structure, noise):

    with open(MGDLfile, 'rb') as f:
        trained_variable, nn_parameter, opt_parameter = pickle.load(f)
        



    train_time = []

    MGDL_time =  0
    MGDL_epochs = 0

    train_loss = []
    validation_loss = []
    train_acc = []
    validation_acc = []

    train_accuracy = []
    validation_accuracy = []




    for grade in range(1, len(nn_parameter["mul_layers_dims"])+1):       
        MGDL_time = MGDL_time + trained_variable['grade'+str(grade)]['train_time']
        MGDL_epochs = MGDL_epochs + opt_parameter['epochs'][grade-1]    

        if grade == 1:
            train_acc.extend(trained_variable['grade'+str(grade)]['train_acc'])
            validation_acc.extend(trained_variable['grade'+str(grade)]['validation_acc'])  
        else:
            train_acc.extend(trained_variable['grade'+str(grade)]['train_acc'][1:])
            validation_acc.extend(trained_variable['grade'+str(grade)]['validation_acc'][1:])  


        train_accuracy.append(trained_variable['grade'+str(grade)]['train_accuracy'])  
        validation_accuracy.append(trained_variable['grade'+str(grade)]['validation_accuracy'])  


    print('MGDL train accuracy for each grade is {}'.format(train_accuracy))
    print('MGDL validation accuracy for each grade is {}'.format(validation_accuracy))
    print('MGDL train time is {}'.format(MGDL_time))

    MGDL_xaxis = np.arange(0, MGDL_epochs)* (MGDL_time / MGDL_epochs)     


    plt.plot(np.array(validation_acc), label='test accuracy')
    plt.plot(np.array(train_acc),  label='train accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    #plt.ylim([0.6, 0.9])    #noise 0.2
    #plt.ylim([0.65, 1.01])    #noise 0.2
    #plt.ylim([0.4, 0.9])    #noise 0.2
    plt.ylim([0.4, 1.01])    #noise 0.2
    plt.legend()
    plt.title('MGDL: $({}, {})$'.format(opt_parameter["max_learning_rate"], opt_parameter["min_learning_rate"]))



    if str(nn_parameter["mul_layers_dims"]) == '[[784, 128, 128, 128, 10], [128, 256, 256, 256, 10], [256, 512, 256, 128, 10]]':
        plt.axvline(x=50, color='k', linestyle=':')
        plt.axvline(x=100, color='k', linestyle=':')
        #plt.axvline(x=600, color='k', linestyle=':')
        plt.xticks([0, 50, 100, 200]) 

    elif str(nn_parameter["mul_layers_dims"]) == '[[784, 128, 10], [128, 128, 10], [128, 128, 10], [128, 256, 10], [256, 256, 10], [256, 256, 10], [256, 512, 10], [512, 256, 10], [256, 128, 10]]':
        plt.axvline(x=10, color='k', linestyle=':')
        plt.axvline(x=20, color='k', linestyle=':')
        plt.axvline(x=40, color='k', linestyle=':')
        plt.axvline(x=60, color='k', linestyle=':')
        plt.axvline(x=80, color='k', linestyle=':')
        plt.axvline(x=110, color='k', linestyle=':')
        plt.axvline(x=140, color='k', linestyle=':')
        plt.axvline(x=170, color='k', linestyle=':')       

        plt.xticks([0, 10, 20, 40, 60, 80, 110, 140, 170, 200])         

    else:
        #Adding vertical dashed lines to separate the grades visually
        plt.axvline(x=10, color='k', linestyle=':')
        plt.axvline(x=30, color='k', linestyle=':')
        plt.axvline(x=60, color='k', linestyle=':')
        plt.xticks([0, 10, 30, 60, 100])

    plt.tight_layout()



    # max_lr_str = '{:.0e}'.format(opt_parameter["max_learning_rate"]).replace('e-0', 'e-')
    # min_lr_str = '{:.0e}'.format(opt_parameter["min_learning_rate"]).replace('e-0', 'e-')
    # fig_filename = 'Fig/MINST_noise{:.1f}_S{}_MGDL_{}{}.png'.format(noise, structure, max_lr_str, min_lr_str)
    # plt.savefig(fig_filename)
    plt.show()

