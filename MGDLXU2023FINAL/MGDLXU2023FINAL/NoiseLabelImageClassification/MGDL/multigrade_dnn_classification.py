import numpy as np
import math
import copy
import time



#-------------------initialize_parameters for grade 1: {W_i, b_i}_{i=1}^{n1}, {M_i}_{i=0}^{n1}--------------------------------
def initialize_parameters_deep(layers_dims, init_method):
    """
    initialize parameter for deep nerual network
    

    Parameters
    ----------
    layers_dims :        list
                         containing the dimensions of each layer in network
                         [input, hiddlen_layer1, ..., hiddlen_layern1, output]
    init_method :        string
                         method of initiation
                         "he": he initalize 

 
    
    Returns
    -------
    parameters :         dictionary 
                         containing neural network parameters
                         Wl : weight matrix of shape (layer_dims[l], layer_dims[l-1])
                         bl : bias vector of shape (layer_dims[l], 1)


                       
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)            # number of layers in the network  
    
        
        
        
    #{W_i, b_i}_{i=1}^{n1}
    for l in range(1, L):

        if init_method == "he":
            print('------------------we use he initialize--------------')
            ## He initialize : Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))   
        elif init_method == "xavier":
            print('------------------we use xavier initialize-------------------')
            #xavier initialize:Understanding the difficulty of training deep feedforward neural networks
            parameters["W" + str(l)] = np.random.uniform(-1, 1, (layers_dims[l], layers_dims[l - 1])) * np.sqrt(1 / layers_dims[l - 1])
            parameters["b" + str(l)] = np.random.uniform(-1, 1, (layers_dims[l], 1)) * np.sqrt(1 / layers_dims[l - 1])



        
    return parameters
#---------------------------------------------------------------------------------------------------------------------------




#------------------------------------------linear_activation_forward---------------------------------------------------------
def linear_activation_forward(N_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer


    Parameters
    ----------
    N_prev :           numpy array of shape (dimension of previous, number of example) 
                       activations from previous layer (or input data)
    W :                numpy array of shape (size of current layer, size of previous layer)
                       weights matrix  
    b :                numpy array of shape (size of the current layer, 1)
                       bias vector 
    activation :       string,  "sin" or "relu" or "identity"
                       the activation to be used in this layer

    Returns
    -------
    N :                numpy array of shape (size of the current layer, 1)
                       the output of the activation function, also called the post-activation value 
    cache :            dictionary containing "linear_cache" and "activation_cache";
                       stored for computing the backward pass efficiently



    """
    Z = np.dot(W, N_prev) + b

    if activation == "relu":
        N = np.maximum(0,Z)
    elif activation == 'softmax':
        N = np.exp(Z)/np.sum(np.exp(Z), axis=0, keepdims=True)
    elif activation == "identity":
        N = Z   

    cache = ((N_prev, W, b), Z)

    return N, cache
#----------------------------------------------------------------------------------------------------------------------------




#-------------------------------------------multigrade model_forward---------------------------------------------------------
def multigrade_forward(X, layers_dims, parameters, grade):
    """
    Implement forward propagation
    
    Parameters
    ----------
    X :                numpy array of shape (input size, number of examples)
                       dataset
    layers_dims :      list 
                       containing the dimensions of each layer in network
    parameters :       dictionary
                       output of initialize_parameters_deep()
    
    Returns
    -------
    N :                numpy array
                       multigrade dnn predict  
    caches :           list
                       containing [((N_0, W_1, b_1)， Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1})]
    
    """

    caches = []
    N = X
    L = len(layers_dims)                     # number of layers in the neural network, W, b
    
    if grade==1:
        activation = "softmax"
    else:
        activation = "identity"
    
    # Implement forward propagation
    for l in range(1, L):
        N_prev = N 
        
        if l == L-1:
            N, cache = linear_activation_forward(N_prev, parameters["W" + str(l)], 
                                                 parameters["b" + str(l)], activation)            
        else:
            N, cache = linear_activation_forward(N_prev, parameters["W" + str(l)], 
                                                 parameters["b" + str(l)], "relu")               

        caches.append(cache)
        
    
                


    return N, caches
#---------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------compute square cost--------------------------------------------------
def squareloss(N, Y):
    """
    Implement the square loss function
                        loss = 1/(2m) sum_{i=1}^{m}(phi(x_i) - y_i)^2

    Parameters
    ----------
    N :                numpy array
                       singlegrade dnn predict  
    Y :                numpy array
                       true "label" vector                  

    Returns
    -------
    loss :              number 
                        square loss
          
    
    """
    # the number of train data 
    m = Y.shape[1]
    #compute square loss
    loss = np.sum( np.square(Y - N) ) / (2*m) 
    
    return loss
#--------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------compute crossentropy loss-------------------------------------------------
def crossentropy(N, Y):
    """
    Implement the cross
     L(s(z), y) =  - 1/N sum_{l=1}^{N} sum_{k=1}^K y_{l, k} log(s_{l, k}), where s(z) is the softmax function of z

    Parameters:
    N :                numpy array
                       singlegrade dnn prediction labels
    Y :                numpy array
                       true label from dataset
                       
    Returns
    -------
    cost :             number 
                       cross entropy loss and L2 regulairzation cost function

    """
    
    m = Y.shape[1]

    # Compute loss from N and y.
    loss = (-1 / m) *np.sum(  np.multiply(Y, np.log(N))  )
    
    
    return loss
#--------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------compute gradient------------------------------------------------------           
#------------------------------------------linear_activation_backward--------------------------------------------------------
'''---------------------compute gradient for {W_i, b_i}_{i=0}^{n1}--------------------------------------------'''

def multigrade_linear_activation_backward(dN, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer. compute gradient for W, b
    
    Parameters
    ----------
    dN :            numpy array with shape (dimension of current layer, number of example) 
                    post-activation gradient for current layer l 
    caches :        list
                    containing [((N_0, W_1, b_1)， Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1}), ((N_{n1}))]
    activation :    string,  "relu" or "softmax"
                    the activation to be used in this layer
    
    Returns
    -------
    dN_prev :       numpy array with shape (dimension of previous layer, number of example)
                    gradient of the cost with respect to the activation (of the previous layer l-1), same shape as N_prev
    dW :            numpy array with shape (dimension of current layer, dimension of previous layer)
                    gradient of the cost with respect to W (current layer l), same shape as W
    db :            numpy array with shape (dimension of current layer, 1)
                    gradient of the cost with respect to b (current layer l), same shape as b
        
        
    """
    linear_cache, activation_cache = cache
    
    
    # calculate dZ
    if activation == "relu":
        dN[activation_cache<0] = 0
        dZ = dN    
    elif activation == "identity":
        dZ = dN
                
        
    #calculate dW, db, dN_prev based on dZ     
    N_prev, W, b = linear_cache

    dW = np.dot(dZ, N_prev.T)       
    db = np.sum(dZ, axis=1, keepdims=True)
    dN_prev = np.dot(W.T, dZ)
    
    return dN_prev, dW, db
#---------------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------------------------
def multigrade_linear_activation_backward_with_softmax(Y, cache, activation="softmax"):
    """
    
    Parameters
    ----------
    Y :             numpy array
                    true label from dataset
    caches :        list
                    containing [((N_0, W_1, b_1)， Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1}), ((N_{n1}))]
    activation :    string,  "softmax"
                    the activation to be used in this layer
                    
    
    Returns
    -------
    dN_prev :       numpy array with shape (dimension of previous layer, number of example)
                    gradient of the cost with respect to the activation (of the previous layer l-1), same shape as N_prev
    dW :            numpy array with shape (dimension of current layer, dimension of previous layer)
                    gradient of the cost with respect to W (current layer l), same shape as W
    db :            numpy array with shape (dimension of current layer, 1)
                    gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache, activation_cache = cache
    
    m = Y.shape[1]
    
    dZ = (np.exp(activation_cache)/np.sum(np.exp(activation_cache), axis=0, keepdims=True) - Y)/m    
    #calculate dW, db, dN_prev based on dZ     
    N_prev, W, b = linear_cache
    dW = np.dot(dZ, N_prev.T)          
    db = np.sum(dZ, axis=1, keepdims=True)
    dN_prev = np.dot(W.T, dZ)

    return dN_prev, dW, db
#---------------------------------------------------------------------------------------------------------------------------



#---------------------------------------compute gradient for W, b-----------------------------------------------------------
def multigrade_backward(Y, N, layers_dims, parameters, caches, grade):
    """
    Implement the backward propagation
    
    Parameters
    ----------
    Y :              numpy array with shape (the dimension of output, number of data)
                     the "label" of data                 
    N :              numpy array with shape (the dimension of output, number of data)
                     predict of the multigrade dnn                                           
    layers_dims :    list 
                     containing the dimensions of each layer in network
    parameters :     dictionary 
                     containing parameters of the model
    caches :         list 
                     containing [((N_0, W_1, b_1), Z_1), ((N_1, W_2, b_2)， Z_2), ..., ((N_{n1-1}, W_{n1}, b_{n1})， Z_{n1}), ((N_{n1}))]
    lambd_W :        scalar
                     regularization hyperparameter for weight W
    activation :    string,  "sin" or "relu"
                    the activation to be used in this layer        

    
    Returns
    -------
    grads :          dictionary with the gradients
                     grads["dN" + str(l)] = ... 
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ... 
                    
                    
    """
    # the number of train data 
    m = Y.shape[1]    
    
    L = len(layers_dims) # the number of layers
    grads = {}
    
    
    # Initializing the backpropagation
    if grade==1:
        current_cache = caches[L-2]
        grads["dN" + str(L-2)], grads["dW" + str(L-1)], grads["db" + str(L-1)] = multigrade_linear_activation_backward_with_softmax(Y, current_cache, "softmax")
    else:
        grads["dN" + str(L-1)] =  1/m * (N - Y)
        grads["dN" + str(L-2)], grads["dW" + str(L-1)], grads["db" + str(L-1)] = multigrade_linear_activation_backward(grads["dN" + str(L-1)], caches[L-2], "identity")
    
    
    for l in reversed(range(1, L-1)):
        # backward
        grads["dN" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = multigrade_linear_activation_backward(grads["dN" + str(l)], caches[l-1], "relu")           
            
    return grads
#---------------------------------------------------------------------------------------------------------------------------




##stop here
#------------------------------------------------------optimization---------------------------------------------------------
"""-----------------------------------------------------random mini batches----------------------------------------------"""
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Parameter
    ---------
    X :                  numpy array with shape (input size, number of examples)      
                         input data 
    Y :                  numpy array with shape (output size, number of examples)              
                         true "label" vector 
    mini_batch_size :    integer 
                         size of the mini-batches
    
    Returns
    -------
    mini_batches :       list 
                         synchronous (mini_batch_X, mini_batch_Y)
                        
                        
    """    
    
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    c, m = Y.shape                  # m, number of training examples, c, number of components
                               
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((c,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
#---------------------------------------------------------------------------------------------------------------------------





"""--------------------------------------------------adam method-------------------------------------------------------"""
#-----------------------------------------------------initialize_adam-----------------------------------------------
def initialize_adam(parameters, layers_dims) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL", "dM0", "dM1", ..., "dML" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Parameters
    ----------
    parameters :         dictionary containing parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
                         parameters["M" + str(l)] = Ml
    layers_dims :        list 
                         containing the dimensions of each layer in network
    
    Returns
    -------
    v :                  dictionary containing the exponentially weighted average of the gradient
                         v["dW" + str(l)] = ...
                         v["db" + str(l)] = ...
    s :                  dictionary containing the exponentially weighted average of the squared gradient.
                         s["dW" + str(l)] = ...
                         s["db" + str(l)] = ...

    
   """
    
    L = len(layers_dims)   # number of layers in the neural networks
    v = {}
    s = {}
    
    
    for l in range(1, L):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])

    
    return v, s
#------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------update_parameters_with_adam-------------------------------------------------
def update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate=0.001,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    
    Parameters
    ----------
    parameters :         dictionary containing parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
    grads :              dictionary containing your gradients for each parameters:
                         grads['dW' + str(l)] = dWl
                         grads['db' + str(l)] = dbl
    v :                  dictionary containing the exponentially weighted average of the gradient
                         v["dW" + str(l)] = ...
                         v["db" + str(l)] = ...
    s :                  dictionary containing the exponentially weighted average of the squared gradient.
                         s["dW" + str(l)] = ...
                         s["db" + str(l)] = ...
    layers_dims :        list 
                         containing the dimensions of each layer in network
    learning_rate :      scalar
                         learning rate
    beta1 :              scalar
                         exponential decay hyperparameter for the first moment estimates 
    beta2 :              scalar
                         exponential decay hyperparameter for the second moment estimates 
    epsilon :            scalar
                         hyperparameter preventing division by zero in Adam updates

    Returns
    -------
    parameters :         dictionary containing updated parameters
                         parameters["W" + str(l)] = Wl
                         parameters["b" + str(l)] = bl
    v :                  dictionary containing the updated exponentially weighted average of the gradient
                         v["dW" + str(l)] = ...
                         v["db" + str(l)] = ...
    s :                  dictionary containing the updated exponentially weighted average of the squared gradient.
                         s["dW" + str(l)] = ...
                         s["db" + str(l)] = ...
                         
    """
    
    L = len(layers_dims)                     # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # perform Adam update on all parameters
    
    for l in range(1, L):
        # moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]


        # compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))
        

        # moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads["dW" + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads["db" + str(l)], 2)       
        
        
        # compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))   
        
        
        # update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s
#-----------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------put all moudel together---------------------------------------------------
def multigrade_dnn_model_1(data, layers_dims, opt_parameter, max_learning_rate, min_learning_rate, epochs, grade):
    """
    Implements a multigrade deep neural network
    
    Parameters
    ----------
    data :                  dictionary 
                            information of orginal data  (train_X, train_Y, test_X, test_Y) 
    layers_dims :           list 
                            containing the dimensions of each layer in network           
    opt_parameter :         dictionary, information of optimization 
                            containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)                            
                                error ----------------- scalar. if relative error of the cost function less than error then stop      
                                mini_batch_size ------- integer, size of a mini batch
                                beta1 ----------------- scalar, exponential decay hyperparameter for the first moment estimates 
                                beta2 ----------------- scalar, exponential decay hyperparameter for the second moment estimates
                                epsilon --------------- scalar, hyperparameter preventing division by zero in Adam updates 
                                epochs ---------------- integer, number of epochs
    max_learning_rate :     scalar
                            max learning rate
    min_learning_rate :     scalar
                            minimum learning rate
    epochs :                integer
                            the number of steps
    
    Returns
    -------
    
    
    
    learning rate at step k :  max_learning_rate / e^{gamma * k}
                               where gamma = 1/epochs ln( max_learning_rate /  min_learning_rate  )
                       
                          
    """
    
    # optimization parameter
    init_method = opt_parameter["init_method"]
    mini_batch_size = opt_parameter["mini_batch_size"] 
    beta1 = opt_parameter["beta1"] 
    beta2 = opt_parameter["beta2"]
    epsilon = opt_parameter["epsilon"] 
    error = opt_parameter["error"]  

                        
    train_acc = []     
    validation_acc = []
    
    train_loss = []
    validation_loss = []
    REC_FRQ_iter = []
    
    
    
    gamma = 1/epochs*  np.log( max_learning_rate /  min_learning_rate  )
    
    
    t = 0                              # initializing the counter required for Adam update
    seed = 1
    
    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims, init_method)

    
    # Initialize
    v, s = initialize_adam(parameters, layers_dims)
    
    start_time = time.time()
    
    # Optimization loop
    for i in range(epochs+1):


        if i%opt_parameter["REC_FRQ"]==0:
            train_predict, _ = multigrade_forward(data["train_X"], layers_dims, parameters, grade)
            validation_predict, _ = multigrade_forward(data["validation_X"], layers_dims, parameters, grade) 
            train_loss.append(crossentropy(train_predict, data["train_Y"]))
            validation_loss.append(crossentropy(validation_predict, data["validation_Y"]))
            train_acc.append(accuracy(data["train_Y"], train_predict))
            validation_acc.append(accuracy(data["validation_Y"], validation_predict))

            REC_FRQ_iter.append(i)
            
            # print('at epoch {}, train loss is {}, train accuracy is {}, validation accuracy is {}'.format(i, train_loss[-1], train_acc[-1], validation_acc[-1]))

        
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        if opt_parameter['SGD']:
            seed = seed + 1
            minibatches = random_mini_batches(data["train_X"], data["train_Y"], mini_batch_size, seed) 
        else:
            minibatches = [(data["train_X"], data["train_Y"])]


        for minibatch in minibatches:    
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            N, caches = multigrade_forward(minibatch_X, layers_dims, parameters, grade)
            
            # Backpropagation
            grads = multigrade_backward(minibatch_Y, N, layers_dims, parameters, caches, grade)

        
        
            # # gradient checking
            # gradient_check_n(layers_dims, parameters, grads, minibatch_X, minibatch_Y, grade, epsilon=1e-7)
        
 
            t = t + 1 # Adam counter
            learning_rate = max_learning_rate / np.exp( gamma * i )           
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate, 
                                                           beta1, beta2,  epsilon)
            
    end_time = time.time()
    elapsed_time = end_time - start_time

    
    #prepare for next grade
    train_predict, train_caches = multigrade_forward(data["train_X"], layers_dims, parameters, grade)
    validation_predict, validation_caches = multigrade_forward(data["validation_X"], layers_dims, parameters, grade)     
    prepare_next_grade = {}
    prepare_next_grade['train_X'] = train_caches[-1][0][0]  
    prepare_next_grade['train_Y'] = data["train_Y"]  - train_predict
    prepare_next_grade['validation_X'] = validation_caches[-1][0][0]
    prepare_next_grade['validation_Y'] = data["validation_Y"]  - validation_predict 
        
    # store history information
    history_dic = {}
    history_dic["train_loss"] = train_loss
    history_dic["validation_loss"] = validation_loss
    history_dic["train_acc"] = train_acc
    history_dic["validation_acc"] = validation_acc 
    history_dic["train_time"] = elapsed_time
    history_dic["REC_FRQ_iter"] = REC_FRQ_iter
    history_dic["parameters"] = parameters
    history_dic["train_predict"] = train_predict
    history_dic["validation_predict"] = validation_predict
    history_dic["train_accuracy"] = accuracy(data["train_Y"], train_predict)
    history_dic["validation_accuracy"] = accuracy(data["validation_Y"], validation_predict)
    
    print('train acc: {}, validation acc: {}'.format(history_dic["train_accuracy"], history_dic["validation_accuracy"]))

    return history_dic, prepare_next_grade
#-----------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------put all moudel together---------------------------------------------------
def multigrade_dnn_model_ell(orgdata, data, layers_dims, opt_parameter, max_learning_rate, min_learning_rate, epochs, grade, train_previous, validation_previous):
    """
    Implements a multigrade deep neural network
    
    Parameters
    ----------
    data :                  dictionary 
                            information of orginal data  (train_X, train_Y, test_X, test_Y) 
    layers_dims :           list 
                            containing the dimensions of each layer in network           
    opt_parameter :         dictionary, information of optimization 
                            containing (optimizer,  learning_rate, mini_batch_size, beta1, beta2, epsilon, error, epochs)                            
                                error ----------------- scalar. if relative error of the cost function less than error then stop      
                                mini_batch_size ------- integer, size of a mini batch
                                beta1 ----------------- scalar, exponential decay hyperparameter for the first moment estimates 
                                beta2 ----------------- scalar, exponential decay hyperparameter for the second moment estimates
                                epsilon --------------- scalar, hyperparameter preventing division by zero in Adam updates 
                                epochs ---------------- integer, number of epochs
    max_learning_rate :     scalar
                            max learning rate
    min_learning_rate :     scalar
                            minimum learning rate
    epochs :                integer
                            the number of steps
    
    Returns
    -------
    
    
    
    learning rate at step k :  max_learning_rate / e^{gamma * k}
                               where gamma = 1/epochs ln( max_learning_rate /  min_learning_rate  )
                       
                          
    """
    
    # optimization parameter
    init_method = opt_parameter["init_method"]
    mini_batch_size = opt_parameter["mini_batch_size"] 
    beta1 = opt_parameter["beta1"] 
    beta2 = opt_parameter["beta2"]
    epsilon = opt_parameter["epsilon"] 
    error = opt_parameter["error"]  

                        
    train_acc = []     
    validation_acc = []
    
    train_loss = []
    validation_loss = []
    REC_FRQ_iter = []
    
    
    
    gamma = 1/epochs*  np.log( max_learning_rate /  min_learning_rate  )
    
    
    t = 0                              # initializing the counter required for Adam update
    seed = 1
    
    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims, init_method)

    
    # Initialize
    v, s = initialize_adam(parameters, layers_dims)
    
    start_time = time.time()
    
    # Optimization loop
    for i in range(epochs+1):


        if i%opt_parameter["REC_FRQ"]==0:
            train_predict, _ = multigrade_forward(data["train_X"], layers_dims, parameters, grade)
            validation_predict, _ = multigrade_forward(data["validation_X"], layers_dims, parameters, grade) 
            
        
            train_loss.append(squareloss(train_predict, data["train_Y"]))
            validation_loss.append(squareloss(validation_predict, data["validation_Y"]))
                
            train_acc.append(accuracy(orgdata["train_Y"], train_predict+train_previous))
            validation_acc.append(accuracy(orgdata["validation_Y"], validation_predict+validation_previous))
            
            REC_FRQ_iter.append(i)
            
        
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        if opt_parameter['SGD']:
            seed = seed + 1
            minibatches = random_mini_batches(data["train_X"], data["train_Y"], mini_batch_size, seed) 
        else:
            minibatches = [(data["train_X"], data["train_Y"])]


        for minibatch in minibatches:    
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            N, caches = multigrade_forward(minibatch_X, layers_dims, parameters, grade)
            
            # Backpropagation
            grads = multigrade_backward(minibatch_Y, N, layers_dims, parameters, caches, grade)

        
        
            # # gradient checking
            # gradient_check_n(layers_dims, parameters, grads, minibatch_X, minibatch_Y, grade, epsilon=1e-7)
        
 
            t = t + 1 # Adam counter
            learning_rate = max_learning_rate / np.exp( gamma * i )           
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, layers_dims, t, learning_rate, 
                                                           beta1, beta2,  epsilon)
            
    end_time = time.time()
    elapsed_time = end_time - start_time

    
    #prepare for next grade
    train_predict, train_caches = multigrade_forward(data["train_X"], layers_dims, parameters, grade)
    validation_predict, validation_caches = multigrade_forward(data["validation_X"], layers_dims, parameters, grade)     
    prepare_next_grade = {}
    prepare_next_grade['train_X'] = train_caches[-1][0][0]  
    prepare_next_grade['train_Y'] = data["train_Y"]  - train_predict
    prepare_next_grade['validation_X'] = validation_caches[-1][0][0]
    prepare_next_grade['validation_Y'] = data["validation_Y"]  - validation_predict 
        
    # store history information
    history_dic = {}
    history_dic["train_loss"] = train_loss
    history_dic["validation_loss"] = validation_loss
    history_dic["train_acc"] = train_acc
    history_dic["validation_acc"] = validation_acc 
    history_dic["train_time"] = elapsed_time
    history_dic["REC_FRQ_iter"] = REC_FRQ_iter
    history_dic["parameters"] = parameters
    history_dic["train_predict"] = train_predict + train_previous
    history_dic["validation_predict"] = validation_predict + validation_previous
    history_dic["train_accuracy"] = accuracy(orgdata["train_Y"], history_dic["train_predict"])
    history_dic["validation_accuracy"] = accuracy(orgdata["validation_Y"], history_dic["validation_predict"])
    
    print('train acc: {}, validation acc: {}'.format(history_dic["train_accuracy"], history_dic["validation_accuracy"]))

    return history_dic, prepare_next_grade



#----------------------------------------------------calculate accuracy----------------------------------------------------------
def accuracy(y_true, y_predict):
    """
    calculate predict accuracy
    

    Parameters
    ----------
    y_true :           numpy array
                       true label
    y_predict :        numpy predict     
                       predict label          

    Returns
    -------
    acc:               scale
                       the predict accuracy
                       

    """
    c, m = y_true.shape                   #c: number of class, n: number of samples
    p = np.zeros((c, m),dtype=int)
    I = np.eye(c) 
    
    # convert probas to 0/1 predictions
    ntrue = 0
    for i in range(0, y_predict.shape[1]):
        max_idx_prediction = np.argmax(y_predict[:, i])
        max_idx_true = np.argmax(y_true[:, i])
        p[:, i] = I[:, max_idx_prediction]
        
        if max_idx_prediction == max_idx_true:
            ntrue = ntrue + 1
    acc = ntrue/m
    
    return acc
#-----------------------------------------------------------------------------------------------------------------------------
    
