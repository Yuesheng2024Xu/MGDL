from multigrade_dnn_main import multi_grade_dnn 


#set which example. choose from 'example1', 'example2', 'example3' 
example = 'example1'   
#noise free or noise case: False (noise free), True (noise case)
noise = False


#if use stochastic method in Adam, then SGD is 'True' and set minibatch size
#if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
SGD = True                                             
# minibatch size
mini_batch = 32                                        


#set structure for MGDL
mul_layers_dims = [[1, 256, 256, 1], [256, 128, 128, 64, 1], [64, 64, 32, 32, 1]]                   # this is the structure for MGDL
#set activation for MGDL for each grade
activation = ["sin", "relu", "relu"]
#set train epoch for each grade
mul_epochs = [20, 20, 40] 

#set max learning rate and min learning rate for each grade
max_learning_rate = [1e-1, 1e-2, 1e-2]                      
min_learning_rate = [1e-4, 1e-4, 1e-4]   

multi_grade_dnn(max_learning_rate, min_learning_rate, mul_layers_dims, mul_epochs, SGD, mini_batch, activation, example, noise)

