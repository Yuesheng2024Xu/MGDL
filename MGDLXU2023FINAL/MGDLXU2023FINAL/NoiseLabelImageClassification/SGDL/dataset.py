import numpy as np
from keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical




def get_mnist(noise):
    
    # Load the MNIST dataset
    (train_X, train_Y), (validation_X, validation_Y) = mnist.load_data()
    cleantrain_Y = train_Y
    
    
    if noise != 0:
        train_Y, actual_noise = noisify_multiclass_symmetric(cleantrain_Y, noise, nb_classes=10)
    

    # Convert labels to one-hot encoding
    train_Y = to_categorical(train_Y)
    validation_Y = to_categorical(validation_Y)

    
    
    # Normalize the images to the range of [0, 1]
    train_X = train_X.reshape(train_X.shape[0], 784)/255
    validation_X = validation_X.reshape(validation_X.shape[0], 784)/255
     
        
    train_X = train_X.T
    train_Y = train_Y.T
    cleantrain_Y = cleantrain_Y.T
    
    validation_X = validation_X.T
    validation_Y = validation_Y.T

    
    data = {}
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['cleantrain_Y'] = cleantrain_Y
    
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y

    
    return data



def get_fashionmnist(noise):
    
    # Load the MNIST dataset
    (train_X, train_Y), (validation_X, validation_Y) = fashion_mnist.load_data()
    cleantrain_Y = train_Y
    
    
    if noise != 0:
        train_Y, actual_noise = noisify_multiclass_symmetric(cleantrain_Y, noise, nb_classes=10)
    

    # Convert labels to one-hot encoding
    train_Y = to_categorical(train_Y)
    validation_Y = to_categorical(validation_Y)

    
    
    # Normalize the images to the range of [0, 1]
    train_X = train_X.reshape(train_X.shape[0], 784)/255
    validation_X = validation_X.reshape(validation_X.shape[0], 784)/255
     
        
    train_X = train_X.T
    train_Y = train_Y.T
    cleantrain_Y = cleantrain_Y.T
    
    validation_X = validation_X.T
    validation_Y = validation_Y.T

    
    data = {}
    data['train_X'] = train_X
    data['train_Y'] = train_Y
    data['cleantrain_Y'] = cleantrain_Y
    
    data['validation_X'] = validation_X
    data['validation_Y'] = validation_Y

    
    return data




# this function is from : Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    
    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)
        new_y[idx] = np.where(flipped == 1)[1][0]

    return new_y


# this function is from : Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
def noisify_multiclass_symmetric(train_Y, noise, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        train_Y_noisy = multiclass_noisify(train_Y, P=P)
        actual_noise = (train_Y_noisy != train_Y).mean()
        
    return train_Y_noisy, actual_noise


