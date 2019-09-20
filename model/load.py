import numpy as np
import keras.models
from keras.models import load_model
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init():
        loaded_model = load_model('model.h5')
        print("Loaded Model from disk")
        
        graph = tf.get_default_graph()
        
        return loaded_model,graph
