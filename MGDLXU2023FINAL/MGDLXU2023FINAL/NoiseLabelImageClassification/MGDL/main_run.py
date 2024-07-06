from multigrade_dnn_main import multi_grade_dnn


#set dataset
dataname = 'MNIST'    # or FashionMNIST
#set noise level
noise = 0.2          # or 0.4


# if use stochastic method in Adam, then SGD is 'True' and set minibatch size
# if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
SGD = True                                                                                                        
# minibatch size 
mini_batch = 128                                                                                    


#set structure for MGDL
structure = 1
mul_layers_dims = [[784, 64, 10], [64, 128, 10], [128, 256, 10], [256, 128, 10]]    

#set max and min learning rate
max_learning_rate = 1e-3
min_learning_rate = 1e-4
mul_epochs = [10, 20, 30, 40]  
#mul_epochs = [5, 5, 5, 5]

multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, noise, structure, dataname)
