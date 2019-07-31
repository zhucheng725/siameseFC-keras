import numpy as np
import random
import keras
from keras import backend as K

def create_base_network(input_shape):
    #conv1
    input = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(96,(11,11), strides = 2, padding = 'valid')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation = 'relu')(x)
    x = keras.layers.MaxPooling2D(pool_size = (3,3),strides=2)(x)
    #conv2
    x = keras.layers.Conv2D(256,(5,5), strides = 1, padding = 'valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation = 'relu')(x)
    x = keras.layers.MaxPooling2D(pool_size = (3,3),strides=2)(x)
    #conv3
    x = keras.layers.Conv2D(192,(3,3), strides = 1, padding = 'valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation = 'relu')(x)
    #conv4
    x = keras.layers.Conv2D(192,(3,3), strides = 1, padding = 'valid')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation = 'relu')(x)
    #conv5
    x = keras.layers.Conv2D(128,(3,3), strides = 1, padding = 'valid')(x)
    x = keras.layers.BatchNormalization()(x)
    return keras.models.Model(input, x)


def cross_correlation(vects):
    x,y  = vects
    try:
        for i in range(int(x.shape[0])): 
            if i == 0:
                x1 = keras.backend.expand_dims(x[i], axis=0)
                y1 = keras.backend.expand_dims(y[i], axis=-1)
                c1 = keras.backend.conv2d(x1, y1, padding='valid', data_format= 'channels_last')
            else:
                x1 = keras.backend.expand_dims(x[i], axis=0)
                y1 = keras.backend.expand_dims(y[i], axis=-1)
                c1 = K.concatenate([c1,keras.backend.conv2d(x1, y1, padding='valid', data_format= 'channels_last')] , axis=0)
        
        val = np.random.random((int(x.shape[0]), 17, 1, 1))#bias
        b = keras.backend.variable(val)
        c1 = c1 + b
    except TypeError:
        val = np.random.random((1, 17, 1, 1))
        b = keras.backend.variable(val)
        x1 = keras.backend.expand_dims(x[0], axis=0)
        y1 = keras.backend.expand_dims(y[0], axis=-1)
        c1 = keras.backend.conv2d(x1, y1, padding='valid', data_format= 'channels_last') + b
    return c1

input_shape_255 = (255,255,3)
input_shape_127 = (127,127,3)


base_network_255 = create_base_network(input_shape_255)
base_network_127 = create_base_network(input_shape_127)
base_network_255.summary()
print('-----------------------------------')
base_network_127.summary()
print('-----------------------------------')
input_a = keras.Input(shape=input_shape_255)
input_b = keras.Input(shape=input_shape_127)
processed_a = base_network_255(input_a)
processed_b = base_network_127(input_b)


score_map = keras.layers.Lambda(cross_correlation, output_shape=(17,17,1))([processed_a, processed_b])
model = keras.models.Model([input_a, input_b], score_map)
model.summary()




