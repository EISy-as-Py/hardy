# import time
# from datetime import datetime
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import (Dense, Conv2D, MaxPool2D,
                          Flatten)
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.models import Model


# Define the base Keras model to use for comparing the different types of plots
def build_model(training_set, validation_set, config_path='./'):
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
    config_path : str
                  string containing the path to the yaml file representing the
                  classifier hyperparameters

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
    # Get the hyperparameters from the cnn_configuration file
    with open(config_path + 'cnn_config.yaml', 'r') as file:
        hparam = yaml.load(file, Loader=yaml.FullLoader)
    ##################################################################
    # Build CNN Model
    kernel = (hparam['kernel_size'][0], hparam['kernel_size'][0])
    input = (hparam['input_shape'][0], hparam['input_shape'][0],
             hparam['input_shape'][1])
    model = Sequential()
    model.add(Conv2D(hparam['filter_size'][0], kernel,
                     activation=hparam['activation'][0],
                     input_shape=input))
    model.add(Conv2D(2*hparam['filter_size'][0], kernel,
                     activation=hparam['activation'][1]))
    model.add(Conv2D(4*hparam['filter_size'][0], kernel,
                     activation=hparam['activation'][2]))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(hparam['num_classes'][0], activation='softmax'))
    #################################################################
    # set up early stopping to automatically interrupt the model when the loss
    # function does not vary for 3 epochs
    callback = callbacks.EarlyStopping(monitor='loss',
                                       patience=hparam['patience'][0])

    #################################################################
    # compile the optimizer and defined the learning function
    model.compile(optimizer=Adam(lr=hparam['learning_rate'][0]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #################################################################
    # Start the learning step and plot the result of the training and
    # validation sets to determine how well the model learned
    history = model.fit(training_set, epochs=hparam['epochs'][0],
                        callbacks=[callback], shuffle=True,
                        validation_data=validation_set)
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
    ###################################
    # loading the configuration file for tuner

    with open(r'./tuner_config.yaml') as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    ####################################
    # Defining input size

    inputs = tf.keras.Input(shape=(50, 50, 3))
    x = inputs

    ####################################
    # extracting parameters from the parameters file
    # and feeding in the tuner

    for i in range(hp.Int('conv_layers', 1, max(param['layers']),
                          default=3)):
        x = tf.keras.layers.Conv2D(
            filters=getattr(hp, param['filters'][0])
            ('filters_', min(param['filters'][1]['values']),
             max(param['filters'][1]['values']), step=4, default=8),
            kernel_size=getattr(hp, param['kernel_size'][0])
            ('kernel_size_' + str(i), min(param['kernel_size'][1]['values']),
             max(param['kernel_size'][1]['values'])),
            activation=getattr(hp, param['activation'][0])
            ('activation_' + str(i), values=param['activation'][1]['values']),
            padding='same')(x)

    if getattr(hp,
               param['pooling'][0])('pooling',
                                    values=param['pooling'][1]['values'])\
            == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # adding in the optimizer
    optimizer = getattr(hp, param['optimizer'][0])('optimizer',
                                                   values=param['optimizer']
                                                   [1]['values'])

    # compiling neural network model
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def feature_map(image, model, classes, size, layer_num=None,
                save=True, log_dir="./", image_path=None):
    '''
    The function outputs the feature map of given layer.

    The function takes image path, model, number of classes, target
    size to ouput the feature maps for a particular neural network
    model.

    Parameter:
    ----------
    image: str or numpy array
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
    save: bool
          if True it saves the feature maps in the log_dir folder
    log_dir: str
             log directory representing the location of logs

    Returns:
    --------

    feature_map: int
                 if layer_num = 'last', feature_map is probability for
                 classfication
    pyplot: matplotlib.pyplot
            if layer_num is int or None, pyplots are generated

    '''
    if isinstance(image, str):
        if image_path:
            img_feature = load_img(image_path + image,
                                   target_size=(size, size))
            img_feature_array = img_to_array(img_feature)
        else:
            print('the path to the image was not provided')
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

        feature_map_layers(img_feature_array, model, list_layer_pos,
                           save, log_dir)

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
        feature_map_layers(img_feature_array, model, list_layer_pos,
                           save, log_dir)
    return


def feature_map_layers(img_feature_array, model, list_layer_pos, save,
                       log_dir):
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
    save: bool
          if True it saves the feature maps in the log_dir folder
    log_dir: str
             log directory representing the location of logs

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

        if save:
            name_feature_map = "_"+str(model.layers[item].name)+"_"
            new_folder_path = "../report/feature_maps/feature_map"
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            ax.savefig(log_dir+new_folder_path+name_feature_map, dpi=100)
    return plt.show()


# =============================================================================
# def hardy_simple_keras(path, epochs=10, plotting=False,
#                        save_threshold=0.95):
#     '''
#     Wrapper for the simple keras model (all functions above this)
#         This way the simple model can be created and run on a preset
#         image folder with one simple function call!
#         Designed to be looped over in the case of the transformation list.
#
#     I suppose we COULD allow for all of the variables herein to be set
#         (and use the defaults as defined in the functions...), but instead
#         I am choosing for now to hard-code some of those decisions.
#
#     Parameters
#     ----------
#     path: str
#           A string containing the path to the files to use for the learning
#
#     epochs: int
#             number of epochs to iterate over. Takes longer but makes better
#             final model.
#
#     plotting:   bool
#                 decision, whether to generate the history plots for the model
#                 using the plot_history function
#
#     save_threshold:     float (from 0 -> 1)
#                         the fraction of "evaluate testing" that must be
#                         correct to trigger a save_model (and save history)
#                         event.
#                         * NOTE: To never save, can simply enter a number >1.
#
#     Hard-Fixed Parameters
#     ---------------------
#     classes: list
#              A list containing strings of the classes the data is divided in.
#              The class name represent the folder name the files are contained
#              in... So just use all folders (which contain images?) in PATH.
#
#     target_size: tuple
#                  A tuple containing the dimentions of the image to be inputt
#                  in the model... For now we will use (50?) 64.
#     split: float
#             A number between 0 and 1 representing which % of the data
#             will compose the validation set. Here, hard-code 0.1?
#     batch_size: int
#                 The number of files to group up into a batch,
#                     for here, we will hard-code "32"
#     color_mode: str
#                 For now, use only "rgb"
#
#     Returns
#     -------
#     loaded_model : model
#                    model that is loaded from the specified location
#
#     '''
#     clock = time.perf_counter()
#     now_str = datetime.now().strftime('%y%m%d-%H-%M')
#     # ^ Used later for saving, and to
#     found_classes = []
#     for item in os.listdir(path):
#         '''
#         Get list of classes via flow-from-directory:
#             Any item in "Path" is a class if it is a folder with .png images
#             inside of it...
#         '''
#         if os.path.isdir(item):
#             # if it's a folder, check inside for png files
#             for file in os.listdir(os.path.join(path, item)):
#                 if '.png' in file:
#                     found_classes.append(item)
#                     break
#                 else:
#                     pass
#         else:
#             pass
#     assert len(found_classes) >= 2, "Could not find folders in " + path
#
#     training_set, validation_set = learning_set(path, split=0.1,
#                                                 target_size=(64, 64),
#                                                 classes=found_classes,
#                                                 batch_size=32,
#                                                 color_mode='rgb')
#
#     testing_set = test_set(path, target_size=(64, 64), classes=found_classes,
#                            batch_size=32, color_mode='rgb')
#
#     # Input shape NxNx3 for rgb? Either hard-code that or do logic...
#     model, model_history = build_model(training_set, validation_set,
#                                        kernel_size=3, epochs=epochs,
#                                        activation=['relu', 'relu', 'relu'],
#                                        input_shape=(64, 64, 3))
#
#     results = evaluate_model(model, testing_set)
#     # ^ What is shape and meaning of results? I need to investigate that...
#     #   Our docstring says that results[1] is the % correct from evaluate()?
#     #   So that is what I'll use for the save question...
#     conf_matrix, report = report_on_metrics(model, testing_set,
#                                             target_names=found_classes)
#     run_time = time.perf_counter() - clock  # Record Run-Time.
#     # Package the results into a single dictionary, to return with function
#     result_dict = {"result": results,
#                    "conf_matrix": conf_matrix,
#                    "report": report,
#                    "run_time": run_time
#                    }
#     if plotting:
#         plot_history(model_history)  # Returns plt.show()?? fig= ??
#     else:
#         pass
#     if results[1] >= save_threshold:
#         '''
#         This determines whether to save the model (and the results? Format?)
#             Note: today_str moved to top of the function.
#         '''
#
#         save_model_name = os.path.join(path, "model_" + now_str)
#         save_result_name = os.path.join(path, "results_" + now_str)
#
#         save_load_model(save_model_name, model=model, save=True)
#         save_load_model(save_result_name, model=result_dict, save=True)
#
#     # Final Return status - We have the model and the result dictionary.
#     # **Does it make sense to return the moedel if we're looping over this?
#     #       For now, sure... In future might ignore that...
#     return model, result_dict, now_str
# =============================================================================
