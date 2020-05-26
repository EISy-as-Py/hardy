import pickle

# import kerastuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import (Dense, Conv2D, MaxPool2D,
                          Flatten)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# from kerastuner.tuners import RandomSearch, BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks


def learning_set(path, split=0.1, target_size=(50, 50),
                 classes=['noisy', 'not_noisy'], batch_size=32,
                 color_mode='grayscale', **kwargs):
    '''
    A funciton that will create an iterator for the files representing the
    learning sets

    Parameters
    ----------
    path: str
          A string containing the path to the files to use for the learning set
    split: float
            A number between 0 and 1 representing which percentage of the data
            will compose the validation set
    target_size: tuple
                 A tuple containing the dimentions of the image to be inputted
                 in the model
    classes: list
             A list containing strings of the classes the data is divided in.
             The class name represent the folder name the files are contained
             in.
    batch_size: int
                The number of files to group up into a batch
    color_mode: str
                Either grayscale or rgb

    Returns
    -------
    training_set:  Keras image iterator
                The training set containg labelled images
    validation_set: Keras image iterator
                The training set containg labelled images
    '''
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


def test_set(path, target_size=(50, 50),
             classes=['noisy', 'not_noisy'], batch_size=32,
             color_mode='grayscale', **kwargs):
    '''
    A funciton that will create an iterator for the files representing the
    test set

    Parameters
    ----------
    path: str
          A string containing the path to the files to use for the test set
    target_size: tuple
                 A tuple containing the dimentions of the image to be inputted
                 in the model
    classes: list
             A list containing strings of the classes the data is divided in.
             The class name represent the folder name the files are contained
             in.
    batch_size: int
                The number of files to group up into a batch
    color_mode: str
                Either grayscale or rgb

    Returns
    -------
    test_set :  Keras image iterator
                The testing set containg labelled images that was not part of
                the learning dataset
    '''
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
    training_set: Keras image directory iterator
                  The set of files that will be used to train the CNN model
    validation_set: Keras image directory iterator
                    The set of files that will be used to validate the trained
                    model after each epoch
    kernel_size: int
                 Integer indicating teh size of the kernel. The resulting
                 kernel will be square
    epochs: int
            Integer indicating the number of epochs the model will be run for.
    activation: list
                A list contianing strings representing the activation function
                of each layer in the model.
    input_shape: tuple
                 A tuple containing the dimentions of the image to be inputted
                 in the model

    Returns
    -------
    model: Keras sequential model
           The trained convolutional neural network
    history: Keras callbacks function
                   A function that retains information of the loss and
                   performance of the training and validation sets in each
                   epoch.
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

    '''
    Functions that returns plot of the performance of the learning set in each
    epoch.

    Parameters
    ----------
    model_history: Keras callbacks function
                   A function that retains information of the loss and
                   performance of the training and validation sets in each
                   epoch.

    Returns
    -------
    fig: matplotlib plot
         A figure containing two plots showing the change in the loss and
         accuracy during the training of the model

    '''
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

    return plt.show()


def evaluate_model(model, testing_set):
    '''
    Function that returns the evaluation of the model based on the performance
    of the testing set previously separated from the rest of the learning
    dataset.

    Parameters
    ----------
    model :  keras sequential model
            the trained model we want to evaluate using a testing set
    testing_set: Keras image directory iterator
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
    A function that prints the result of the model just trained

    Parameters
    ----------
    model: Keras sequential model
           the trained convolutional neural network
    test_set: Keras image directory iterator
              the test set to use to obtain the true performance of the model.
    target_names: list
                  list containing strings represnting the classes the data
                  is classified in

    Returns
    -------
    conf_matrix : array
                  A numpy array containing values for the true positives,
                  false negatives, false positives and true negatives
    report : str
             a string containg the overall report of the performance
             of the model. Accuracy, recall and F1 scores are reported.
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
        return 'the model was correctly saved'
    elif load:
        loaded_model = pickle.load(open(filename+'.sav', 'rb'))
        return loaded_model


def build_tuner_model(hp):
    '''
    Functions that builds a convolutional keras model with
    tunable hyperparameters


    Parameters
    ----------
    hp: keras tuner class
        A class that is used to define the parameter search space

    Returns
    -------
    model: Keras sequential model
           The trained convolutional neural network
    '''
    inputs = tf.keras.Input(shape=(50, 50, 3))
    x = inputs
    for i in range(hp.Int('conv_layers', 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=4 * i,
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 7),
            activation=hp.Choice('activation_' + str(i),
                                 ['relu', 'sigmoid']))(x)

    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(optimizer, leraning_rate=learning_rate,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# def run_tuner():
#
#     return
