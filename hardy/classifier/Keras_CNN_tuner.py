import keras
from keras.models import Sequential
from tensorflow.keras import layers
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch, BayesianOptimization
import kerastuner as kt
import tensorflow as tf


def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(50, 50, 3))
    x = inputs
    for i in range(hp.Int('conv_layers', 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=4 * i,
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 7),
            activation=hp.Choice('activation_' + str(i),
                                 ['relu', 'sigmoid']))(x)

#     if hp.Choice('global_pooling', ['max', 'avg']) == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
#     else:
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_Rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(optimizer, leraning_rate=learning_rate,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
