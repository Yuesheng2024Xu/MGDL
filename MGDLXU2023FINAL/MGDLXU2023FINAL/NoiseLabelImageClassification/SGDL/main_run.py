from singlegrade_dnn_main import single_dnn_main 


#set dataset
dataname = 'MNIST'    # or FashionMNIST
#set noise level
noise = 0.2          # or 0.4

# if use stochastic method in Adam, then SGD is 'True' and set minibatch size
# if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
SGD = True                                                                                                        
# minibatch size 
mini_batch_size = 128                                                                                    

#set dataset
dataname = 'MNIST'    # or FashionMNIST

structure = 1
layers_dims = [784, 64, 128, 256, 128, 10]                                                           # this is the structure for SGDL

max_learning_rate = 1e-3
min_learning_rate = 1e-4
epochs = 100                                                                                          # the number training epoch in each grade

single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, noise, structure, dataname)

