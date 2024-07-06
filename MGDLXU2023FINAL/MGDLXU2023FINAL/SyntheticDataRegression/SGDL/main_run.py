from singlegrade_dnn_main import single_dnn_main 


#set which example. choose from 'example1', 'example2', 'example3' 
example = 'example1'   
#noise free or noise case: False (noise free), True (noise case)
noise = False


#if use stochastic method in Adam, then SGD is 'True' and set minibatch size
#if use Full grade in Adam, the SGD is 'False' and set mini_batch size to 'False'
SGD = True                                             
# minibatch size
mini_batch_size = 32   

  

#set structure for SGDL
layers_dims = [1, 256, 256, 128, 128, 64, 64, 32, 32, 1]                                                # this is the structure for SGDL
#set train epoch
epochs = 5
#epochs = 550
                                        
#set max learning rate and min learning rate 
max_learning_rate = 1e-3                                                                          # the maximum learning rate, denote as t_max in the paper
min_learning_rate = 1e-4                                                                          # the minimum learning rate, denote as t_min in the paper

single_dnn_main(layers_dims, max_learning_rate, min_learning_rate, epochs, mini_batch_size, SGD, example, noise)

