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
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.models import Model


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


def save_load_model(filepath, model=None, save=None, load=None):
    '''Function to save and load the NN model

    Function that can save or load model depending on given parameters.

    Parameters
    ----------
    filename : str
               string indicating the filename for saving or loading model.
    network : neural_network
              trained neural network variable that is to be saved or loaded.
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
        model.save(filepath)
        return 'the model was correctly saved'
    elif load:
        loaded_model = tf.keras.models.load_model(filepath)
        return loaded_model


def feature_map(image, model, classes, size, layer_num=None):
    '''
    The function outputs the feature map of given layer.

    The function takes image path, model, number of classes, target
    size to ouput the feature maps for a particular neural network
    model.

    Parameter:
    ----------
    imag_path: str or numpy array
               if string it opens the image from path provided. If
               numpy array, it directly feeds it into feature maps
    model: neural network model
           trained neural network model to make prediction
    classes: int
             number of classes used to train the model
    size: int
          target size used to train the model
    layer_num: int or str
               if int, provides output only from a single layer. If
               None, provides output from all the layers. If 'last',
               it provides provides probablity for classifications.

    Returns:
    --------

    feature_map: int
                 if layer_num = 'last', feature_map is probability for
                 classfication
    pyplot: matplotlib.pyplot
            if layer_num is int or None, pyplots are generated

    '''
    if isinstance(image, str):
        img_feature = load_img(image, target_size=(size, size))
        img_feature_array = img_to_array(img_feature)
    else:
        img_feature_array = image

    img_feature_array = expand_dims(img_feature_array, axis=0)

    list_layer_pos = []
    if layer_num is None:
        for i in range(len(model.layers)):
            layer = model.layers[i]
            if 'flatten' in layer.name or layer.output.shape[1] == classes:
                continue
            list_layer_pos.append(i)

        feature_map_layers(img_feature_array, model, list_layer_pos)

    elif layer_num == 'last':
        for i in range(len(model.layers)):
            list_layer_pos.append(i)
        feature_map_model = Model(inputs=model.inputs,
                                  outputs=model.layers[max(list_layer_pos)]
                                  .output)
        feature_map = feature_map_model.predict(img_feature_array)
        print('The output from final layer is {}'.format(feature_map))

    else:
        list_layer_pos.append(layer_num)
        feature_map_layers(img_feature_array, model, list_layer_pos)
    return


def feature_map_layers(img_feature_array, model, list_layer_pos):
    '''
    Nested function for feature_map(). Returns the pyplots for if layer_num
    is int or None in feature_map().

    Parameters:
    -----------

    image_feature_array: array
                         array in expanded dimension representing the image
                         input in feature_map()
    model: neural network model
           neural network model used to make prediction for the image
    list_layer_pos: list
                    list comprising of numbers representing the layer position

    Returns:
    --------

    pyplot: matplotlib.pyplot
            pyplot representing the feature maps

    '''

    for item in list_layer_pos:
        feature_map_model = Model(inputs=model.inputs,
                                  outputs=model.layers[item].output)
        feature_map = feature_map_model.predict(img_feature_array)

        print('The output is from layer {}, {} with \
              shape {}'.format(item, model.layers[item].name,
                               model.layers[item].output.shape))
        ax = plt.figure(figsize=(10, 10))
        for x in range(1, feature_map.shape[3]+1):
            b = ax.add_subplot(6, 6, x)
            b.axis('off')
            plt.imshow(feature_map[0, :, :, x-1], cmap='gray')
    return plt.show()
