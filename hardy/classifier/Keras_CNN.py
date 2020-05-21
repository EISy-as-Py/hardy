import os
import pickle
import random
import shutil
import sys

import kerastuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import (Dense, Dropout, Conv2D, MaxPool2D,
                          Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import RandomSearch, BayesianOptimization
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import metrics
from tensorflow.keras import callbacks


def learning_set(path, split=0.1, target_size=(50, 50),
                 classes=['noisy', 'not_noisy'], batch_size=32,
                 color_mode='grayscale', **kwargs):

    data = ImageDataGenerator(validation_split=split, **kwargs)
    training_set = data.flow_from_directory(path, target_size=target_size,
                                            classes=classes,
                                            batch_size=batch_size,
                                            subset='training', shuffle=True,
                                            color_mode=color_mode)
    validation_set = data.flow_from_directory(path, target_size=target_size,
                                              classes=classes,
                                              batch_size=batch_size,
                                              subset='validation',
                                              shuffle=True,
                                              color_mode=color_mode)

    return training_set, validation_set


def test_set(path, split=0.1, target_size=(50, 50),
             classes=['noisy', 'not_noisy'], batch_size=32,
             color_mode='grayscale', **kwargs):

    data = ImageDataGenerator(**kwargs)
    test_set = data.flow_from_directory(path, target_size=target_size,
                                        classes=classes, batch_size=batch_size,
                                        shuffle=False, color_mode=color_mode)
    return test_set


# Define the base Keras model to use for comparing the different types of plots
def build_model(training_set, validation_set, kernel_size=3, epochs=10,
                activation=['relu', 'relu', 'relu'], input_shape=(50, 50, 1)):
    '''
    Function that allows to build and fit a sequential convolutional
    neural network using Keras.

    Parameters
    ----------

    Returns
    -------


    '''
    #################################################################
    # Build CNN Model
    kernel = (kernel_size, kernel_size)
    model = Sequential()
    model.add(Conv2D(8, kernel, activation=activation[0],
                     input_shape=input_shape))
    model.add(Conv2D(16, kernel, activation=activation[1]))
    model.add(Conv2D(32, kernel, activation=activation[2]))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    #################################################################
    # set up early stopping to automatically interrupt the model when the loss
    # function does not vary for 3 epochs
    callback = callbacks.EarlyStopping(monitor='loss', patience=2)

    #################################################################
    # compile the optimizer and defined the learning function
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #################################################################
    # Start the learning step and plot the result of the training and
    # validation sets to determine how well the model learned
    history = model.fit(training_set, epochs=epochs, callbacks=[callback],
                        shuffle=True, validation_data=validation_set)
    #################################################################

    return model, history


def plot_history(model_history):
    # Let's plot the results
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.5)
    # The Loss function
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(loss)+1)

    ax[0].plot(epochs, loss, 'bo', label='Training_loss')
    ax[0].plot(epochs, val_loss, 'b', label='Validation_loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('CNN Loss per Epoch')

    # The model accuracy
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    ax[1].plot(epochs, acc, 'bo', label='Training_acc')
    ax[1].plot(epochs, val_acc, 'b', label='Validation_acc')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('CNN Accuracy per Epoch')

    return


def evaluate_model(model, testing_set):
    '''
    Function that returns the evaluation of the model based on the performance
    of the testing set previously separated from the rest of the learning
    dataset.

    Parameters
    ----------
    model :  keras model ####
            the trained model we want to evaluate using a testing set
    testing_set: ######
                The testing set containg labelled images that was not part of
                the learning dataset. This will be used to evaluate the actual
                performance of the trained model.
    Returns
    -------
    results[1] : float32
                 returns the classification accuracy of the model based on its
                 performance on the testing set
    '''
    results = model.evaluate(testing_set)
    name = model.metrics_names

    print('\n{} = {:.4f}\n'. format(name[0], results[0]))
    print('{} = {:.4f}\n'. format(name[1], results[1]))

    return results


def report_on_metrics(model, test_set, target_names=['noisy', 'not_noisy']):
    '''

    '''
    Y_pred = model.predict_generator(test_set, len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix \n')
    conf_matrix = confusion_matrix(test_set.classes, y_pred)
    print(conf_matrix)
    print('\n Classification Report')
    report = classification_report(test_set.classes, y_pred,
                                   target_names=target_names)
    print(report)

    return conf_matrix, report


def save_load_model(filename, model=None, save=None, load=None):
    '''Function to save and load the NN model

    Function that can save or load model depending on given parameters.

    Parameters
    ----------
    filename : str
               string indicating the filename for saving or loading model.
    network : neural_network
              neural network variable that is to be saved or loaded.
    save : bool
           boolean value if true saves the neural network model.
    load : bool
           boolean value if true loads the neural network model.

    Returns
    -------
    loaded_model : model
                   model that is loaded from the specified location
    '''
    if save:
        pickle.dump(model, open(filename+'.sav', 'wb'))
        return 0
    elif load:
        loaded_model = pickle.load(open(filename+'.sav', 'rb'))
        return loaded_model

#
# def cross_validation(k, path, split=0.2, target_size=(50, 50),
#                      classes=['noisy', 'not_noisy'], batch_size=32,
#                      color_mode='rgb', kernel_size=3, epochs=5):
#     training_set, validation_set = learning_set(path, split=split,
#                                                 target_size=target_size,
#                                               classes=['noisy', 'not_noisy'],
#                                                 batch_size=batch_size,
#                                                 color_mode='rgb'):
#
#
#     return


def build_tuner_model(hp):
    '''Builds a convolutional keras model with tunable hyperparameters


    Parameters
    ----------


    Returns
    -------

    '''
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
