import keras
import os
import yaml

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hardy.handling import to_catalogue

from keras.layers import (Dense, Conv2D, Flatten)
from keras.models import Sequential
# from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from tensorflow.keras.models import Model


# Define the base Keras model to use for comparing the different types of plots
def build_model(training_set, validation_set=None, config_path='./'):
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
    for i in range(hparam['layers'][0]):
        model.add(Conv2D(np.power(2, i)*hparam['filter_size'][0], kernel,
                         activation=hparam['activation'][i],
                         input_shape=input))

    model.add(getattr(keras.layers, hparam['pooling'][0])(2, 2))
    model.add(Flatten())
    model.add(Dense(hparam['num_classes'][0], activation='softmax'))
    #################################################################
    # set up early stopping to automatically interrupt the model when the loss
    # function does not vary for 3 epochs
    callback = callbacks.EarlyStopping(monitor='loss',
                                       patience=hparam['patience'][0])

    #################################################################
    # compile the optimizer and defined the learning function
    model.compile(getattr(keras.optimizers, hparam['optimizer'][0])(
                  lr=hparam['learning_rate'][0]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #################################################################
    # Start the learning step and plot the result of the training and
    # validation sets to determine how well the model learned
    if validation_set:
        history = model.fit(training_set, epochs=hparam['epochs'][0],
                            callbacks=[callback], shuffle=True,
                            validation_data=validation_set, verbose=2)
    else:
        history = model.fit(training_set, epochs=hparam['epochs'][0],
                            callbacks=[callback], shuffle=True, verbose=2)
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

    return ax


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
    test_set.reset()
    Y_pred = model.predict_generator(test_set, len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix \n')
    if isinstance(test_set, keras.preprocessing.image.DirectoryIterator):
        conf_matrix = confusion_matrix(test_set.classes, y_pred)
        report = classification_report(test_set.classes, y_pred,
                                       target_names=target_names)
    else:
        conf_matrix = confusion_matrix(np.argmax(test_set.y, axis=1), y_pred)
        report = classification_report(np.argmax(test_set.y, axis=1), y_pred)

    print(conf_matrix)
    print('\n Classification Report')

    print(report)

    return conf_matrix, report


def save_load_model(filepath, model=None, save=None, load=None):
    '''Function to save and load the NN model

    Function that can save or load model depending on given parameters.

    Parameters
    ----------
    filepath : str
               string indicating the filename for saving or loading model.
    model : neural_network
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


def feature_map(image, model, classes, size, layer_num=None,
                save=True, log_dir="./", image_path=None):
    '''
    The function outputs the feature map of given layer.

    The function takes image path, model, number of classes, target
    size to ouput the feature maps for a particular neural network
    model.

    Parameters
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

    Returns
    -------

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
            elif 'global' in layer.name:
                continue
            list_layer_pos.append(i)

        return feature_map_layers(img_feature_array, model, list_layer_pos,
                                  save, log_dir)

    elif layer_num == 'last':
        feature_map_model = model
        # for i in range(len(model.layers)):
        #     list_layer_pos.append(i)
        # feature_map_model = Model(inputs=model.inputs,
        #                         outputs=model.layers[max(list_layer_pos)]
        #                         .output)
        feature_map = feature_map_model.predict(img_feature_array)
        print('The output from final layer is {}'.format(feature_map))
        return feature_map
    else:
        list_layer_pos.append(layer_num)
        return feature_map_layers(img_feature_array, model, list_layer_pos,
                                  save, log_dir)


def feature_map_layers(img_feature_array, model, list_layer_pos, save,
                       log_dir):
    '''
    Nested function for feature_map(). Returns the pyplots for if layer_num
    is int or None in feature_map().

    Parameters
    ----------

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

    Returns
    -------

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
        if feature_map_model.layers[item].output.shape[3]/6 <= 6:
            rows = 6
        else:
            rows = feature_map_model.layers[item].output.shape[3]/6
        for x in range(1, feature_map.shape[3]+1):
            b = ax.add_subplot(int(rows), 6, x)
            b.axis('off')
            plt.imshow(feature_map[0, :, :, x-1], cmap='gray')

        if save:
            name_feature_map = "feature_map_"+str(model.layers[item].name)+"_"
            new_folder_path = "/report/feature_maps/"
            if not os.path.exists(log_dir+new_folder_path):
                os.makedirs(log_dir+new_folder_path)
            ax.savefig(log_dir+new_folder_path+name_feature_map, dpi=100)
    return ax


def k_fold_model(k, config_path='./', target_size=(80, 80),
                 classes=['noisy', 'not_noisy'], batch_size=32,
                 color_mode='rgb', iterator_mode='arrays',
                 image_list=None, test_set=None, **kwargs):
    '''

    '''

    validation_score = []

    for fold in range(k):
        train_data, val_data = to_catalogue.learning_set(
            target_size=target_size, classes=classes, batch_size=batch_size,
            color_mode=color_mode, iterator_mode='arrays',
            image_list=image_list, k_fold=True, k=k, fold=fold, **kwargs)
        model, history = build_model(train_data, config_path=config_path)
        validation_score.append(evaluate_model(model, val_data)[1])

    validation_score = np.average(validation_score)
    print('The average model accuracy is {} for {} number of folds'.format(
        np.round(validation_score, 3), k))

    # Retrain the model with the entirety of the data set
    # and return its performance
    train_data, val_data = to_catalogue.learning_set(
        target_size=target_size, classes=classes, batch_size=batch_size,
        color_mode=color_mode, iterator_mode='arrays', split=0,
        image_list=image_list, **kwargs)
    model, history = build_model(train_data, config_path=config_path)
    final_score = evaluate_model(model, test_set)

    print('The final model accuracy is {}'.format(final_score[1]))

    return validation_score, model, history, final_score
