import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def example1(opt):
    """
    np.sin(100 * x)
    """
    np.random.seed(0)
    #target function
    def f(x):
        return np.sin(100 * x)
    
    #--------------train data----------------------
    train_X = np.linspace(0, 1,  opt.ntrain)
    if opt.noise:
        en = np.random.normal(0, 0.05,  opt.ntrain)
    else:
        en = 0
    train_Y = f(train_X) + en

    
    #------------test data------------------------
    test_X = np.linspace(0, 1,opt.ntest)
    test_Y = f(test_X)

    #-----------validation data-------------------
    nval = int(0.2 *  opt.ntrain)
    indices = np.random.choice( opt.ntrain, nval, replace=False)
    validation_X = train_X[indices]
    validation_Y = train_Y[indices] + np.random.normal(0, 0.01, validation_X.shape)
    
    
    data = {}
    data['train_X'] = train_X.reshape((1, train_X.shape[0]))
    data['train_Y'] = train_Y.reshape((1, train_Y.shape[0]))
    data['validation_X'] = validation_X.reshape((1, validation_X.shape[0]))
    data['validation_Y'] = validation_Y.reshape((1, validation_Y.shape[0]))
    data['test_X'] = test_X.reshape((1, test_X.shape[0]))
    data['test_Y'] = test_Y.reshape((1, test_Y.shape[0]))
    
    
    return data




def example2(opt):
    """
    x*np.sin(100 * x)
    """
    np.random.seed(0)
    #target function
    def f(x):
        return x * np.sin(100 * x)
    
    #--------------train data----------------------
    train_X = np.linspace(0, 1,  opt.ntrain)
    if opt.noise:
        en = np.random.normal(0, 0.05,  opt.ntrain)
    else:
        en = 0
    train_Y = f(train_X) + en

    
    #------------test data------------------------
    test_X = np.linspace(0, 1,opt.ntest)
    test_Y = f(test_X)

    #-----------validation data-------------------
    nval = int(0.2 *  opt.ntrain)
    indices = np.random.choice( opt.ntrain, nval, replace=False)
    validation_X = train_X[indices]
    validation_Y = train_Y[indices] + np.random.normal(0, 0.01, validation_X.shape)
    
    
    data = {}
    data['train_X'] = train_X.reshape((1, train_X.shape[0]))
    data['train_Y'] = train_Y.reshape((1, train_Y.shape[0]))
    data['validation_X'] = validation_X.reshape((1, validation_X.shape[0]))
    data['validation_Y'] = validation_Y.reshape((1, validation_Y.shape[0]))
    data['test_X'] = test_X.reshape((1, test_X.shape[0]))
    data['test_Y'] = test_Y.reshape((1, test_Y.shape[0]))
    
    
    return data



def example3(opt):
    np.random.seed(0)
    #target function
    def f(x):
        fx = np.abs(np.cos(15 * np.pi * (x - 0.3)) - 0.7)
        fx = np.abs(np.cos(2 * np.pi * (fx - 0.5)) - 0.5)
        fx = -np.abs(fx - 1.3) + 1.3
        fx = -np.abs(fx - 0.9) + 0.9
        return (x + 1) * fx
    
    #--------------train data----------------------
    train_X = np.linspace(-1, 1,  opt.ntrain)
    if opt.noise:
        en = np.random.normal(0, 0.05,  opt.ntrain)
    else:
        en = 0
    train_Y = f(train_X) + en

    
    #------------test data------------------------
    test_X = np.linspace(-1, 1,opt.ntest)
    test_Y = f(test_X)

    #-----------validation data-------------------
    nval = int(0.2 *  opt.ntrain)
    indices = np.random.choice( opt.ntrain, nval, replace=False)
    validation_X = train_X[indices]
    validation_Y = train_Y[indices] + np.random.normal(0, 0.01, validation_X.shape)
    
    
    data = {}
    data['train_X'] = train_X.reshape((1, train_X.shape[0]))
    data['train_Y'] = train_Y.reshape((1, train_Y.shape[0]))
    data['validation_X'] = validation_X.reshape((1, validation_X.shape[0]))
    data['validation_Y'] = validation_Y.reshape((1, validation_Y.shape[0]))
    data['test_X'] = test_X.reshape((1, test_X.shape[0]))
    data['test_Y'] = test_Y.reshape((1, test_Y.shape[0]))
    
    
    return data


