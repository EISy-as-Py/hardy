import numpy as np
import pandas as pd
import pickle
import os

import hardy.handling.visualization as vis
import hardy.handling as handling
from keras.preprocessing.image import ImageDataGenerator
import keras


def save_load_data(filename, data=None, save=None, load=None,
                   file_extension='.sav', location='./'):
    """Function to save and load model

    Function that can save or load model depending on given parameters.

    Parameters
    ----------
    filename : str
               string indicating the filename for saving or loading dataset.
    data : list
           dataset that is to be saved or loaded.
    save : bool
           boolean value if true saves the compressed dataset.
    load : bool
           boolean value if true loads the compressed dataset.
    file_extension : str
                     String containing the file extension to use
    location :  str
                string containing the path to the folder to save the
                pickled file in

    Returns
    -------
    loaded_data : list
                  dataset that is loaded from the specified location
    """
    if save:
        pickle.dump(data, open(location + filename + file_extension, 'wb'))
        return print('Successfully Pickled')
    elif load:
        loaded_data = pickle.load(open(location + filename + file_extension,
                                  'rb'))
        return loaded_data


def rgb_list(input_path='./', skiprows=6, df_list=None, serial_list=None,
             classes=['noisy', 'not_noisy'],
             plot_format='single',  column_names=None,
             combine_method='add'):
    '''
    Input a path of csv files (with some guidance),
    Plot them RGB-wise into images
    return a list of tuples as to be fed into the keras PreProcess f(n)

    INPUTS:


    RETURNS

    list_of_tuples  :   list of tuples, following: (SERIAL, IMG, LABEL)

        SERIAL      :   File name with the extension taken off
                            (We should parse with . not just [-4])

        IMG         :   ndarray of NxNx3

    '''
    list_of_tuples = []
    if not classes:
        classes = handling.cats_from_fnames(os.listdir(input_path))

    for entry in os.listdir(input_path):
        if entry.endswith('.csv'):
            # Read data into pandas dataframe
            fdata = pd.read_csv(input_path+entry, skiprows=skiprows)

            rgb_image = rgb_visualize(fdata, plot_format, combine_method,
                                      column_names)

#  The labelling of the data is somewhat hardcoded in this funciton right now.
#  Consider improving it. We can now call cats_from_fnames for the full list.
#            if classes[0] in entry:
#                label = classes[0]
#            else:
#                label = classes[1]

            label = None
            for each_label in classes:
                # Find the first label that matches.
                if not label and each_label in entry:
                    label = each_label
                else:
                    pass
            if not label:
                # If none of the labels fit, make new "not" label
                label = "not_" + classes[0]
            else:
                pass
            list_of_tuples.append((entry.rstrip(entry[-4:]),
                                  rgb_image, label))

    return list_of_tuples


def data_set_split(image_list, test_set_filenames):
    '''
    Function that splits the list of image arrays into a test set and a
    learning setto use for the classification step

    Parameters
    ----------
    image_list : list
                 A list of tuples containing the filenames, the arrays
                 reoresenitng the images and their labels
    test_set_filenames : list
                         List of strings containig the filename of the datasets
                         selected to the be in the test set

    Returns
    -------
    test_set_list : list
                    A list of tuples containing the filenames, the arrays
                    reoresenitng the images and their labels to be used as
                    the test set
    learning_set_list : list
                        A list of tuples containing the filenames, the arrays
                        reoresenitng the images and their labels to be used as
                        the learning set

    '''
    test_set_list = [n for n in image_list if n[0][:][:] in test_set_filenames]
    learning_set_list = [n for n in image_list if n not in test_set_list]

    return test_set_list, learning_set_list


def rgb_visualize(fdata, plot_format='RGBrgb', combine_method='add',
                  column_names=None):
    '''
        Input a list of dataframes (already read and/or processed),
        Plot them RGB-wise into images
        return a list of tuples as to be fed into the keras PreProcess f(n)

        INPUTS:
            plot_format :   EITHER 'single' (bodge, depreciate later)
                            OR some combination of "RGBrgb", which will be
                            the order of columns plotted:
                            R = red   X-axis      r = red   Y-axis
                            G = green X-axis      g = green Y-axis
                            B = blue  X-axis      b = blue   Y-axis
                             * X = do not plot (skip column)
                             ** If RGBrgb letters are missing, simply pass
                                to the plotting function as "None"

                            The to-be-depreciated 'single' is thus:
                                "RB"
                            The As-written "else" is thus:
                                "Rb"

            combine_method: "add" or "mlt" - which visualization fn to use

        RETURNS

        list_of_tuples  :   list of tuples, following: (SERIAL, IMG, LABEL)

            SERIAL      :   File name with the extension taken off
                                (We should parse with . not just [-4])

            IMG         :   ndarray of NxNx3

    '''
    if not column_names:
        column_names = fdata.columns.keys()

    if plot_format == 'single':
        rgb_image = vis.rgb_plot(red_array=fdata[column_names[0]],
                                 blue_array=fdata[column_names[1]],
                                 plot=False)
    elif plot_format == "else":
        rgb_image_x = vis.rgb_plot(red_array=fdata[column_names[0]],
                                   plot=False)
        rgb_image_y = vis.rgb_plot(blue_array=fdata[column_names[1]],
                                   plot=False)
        rgb_image = vis.orthogonal_images_add(rgb_image_x, rgb_image_y,
                                              plot=False)
    else:
        # Writing new Decision-matrix to organize with the input-string
        # Loop through the string, and if you see an "RGB,rgb",
        #   then that column is the one which will go there!
        R = None
        G = None
        B = None
        r = None
        g = None
        b = None
        for i in range(len(plot_format)):
            # Loop through the string. react to FIRST encounter of str
            if not R and plot_format[i] == "R":
                R = fdata[column_names[i]]
            if not G and plot_format[i] == "G":
                G = fdata[column_names[i]]
            if not B and plot_format[i] == "B":
                B = fdata[column_names[i]]
            if not r and plot_format[i] == "r":
                r = fdata[column_names[i]]
            if not g and plot_format[i] == "g":
                g = fdata[column_names[i]]
            if not b and plot_format[i] == "b":
                b = fdata[column_names[i]]
        rgb_image_x = vis.rgb_plot(red_array=R, green_array=G,
                                   blue_array=B, plot=False)
        rgb_image_y = vis.rgb_plot(red_array=r, green_array=g,
                                   blue_array=b, plot=False)

        # Default to "Add", but check for the option of using the mlt fn.
        if combine_method == "mlt":
            rgb_image = vis.orthogonal_images_mlt(rgb_image_x,
                                                  rgb_image_y,
                                                  plot=False)
        else:
            rgb_image = vis.orthogonal_images_add(rgb_image_x,
                                                  rgb_image_y,
                                                  plot=False)
    return rgb_image

############################################################################
# Generating the sets to use for the classification step


def learning_set(path=None, split=0.1, target_size=(80, 80),
                 classes=['noisy', 'not_noisy'], batch_size=32,
                 color_mode='rgb', iterator_mode='arrays',
                 image_list=None, **kwargs):
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
    iterator_mode : str
    image_list : list

    Returns
    -------
    training_set:  Keras image iterator
                The training set containg labelled images
    validation_set: Keras image iterator
                The training set containg labelled images
    '''
    data = ImageDataGenerator(validation_split=split, **kwargs)

    if iterator_mode == 'arrays':
        n = target_size[0]
        if color_mode == 'rgb':
            channels = 3
        else:
            channels = 1

        assert image_list, 'the image arrays should be provided'
# Add checks for the image arrays- (filename, arrays, label)
# assert im
        image_arrays = np.array([image_list[i][1][:]
                                for i in range(len(image_list))])
        image_data = image_arrays.reshape(image_arrays.shape[0], n,
                                          n, channels).astype('float32')
        image_data = (image_data*255).astype('uint8')
        image_labels = np.array([image_list[i][:][2]
                                 for i in range(len(image_list))])
        image_labels = keras.utils.to_categorical(image_labels, num_classes=2)

        training_set = data.flow(x=image_data, y=image_labels,
                                 batch_size=batch_size, subset='training')
        validation_set = data.flow(x=image_data, y=image_labels,
                                   batch_size=batch_size, subset='validation')

    else:
        training_set = data.flow_from_directory(path,
                                                target_size=target_size,
                                                classes=classes,
                                                batch_size=batch_size,
                                                subset='training',
                                                shuffle=True,
                                                color_mode=color_mode)
        validation_set = data.flow_from_directory(path,
                                                  target_size=target_size,
                                                  classes=classes,
                                                  batch_size=batch_size,
                                                  subset='validation',
                                                  shuffle=True,
                                                  color_mode=color_mode)

    return training_set, validation_set


def test_set(path, target_size=(80, 80),
             classes=['noisy', 'not_noisy'], batch_size=32,
             color_mode='rgb', iterator_mode='arrays',
             image_list=None, **kwargs):
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
    if iterator_mode == 'arrays':
        n = target_size[0]
        if color_mode == 'rgb':
            channels = 3
        else:
            channels = 1

        assert image_list, 'the image arrays should be provided'
# Add checks for the image arrays- (filename, arrays, label)
# assert im
        image_arrays = np.array([image_list[i][1][:]
                                for i in range(len(image_list))])
        image_data = image_arrays.reshape(image_arrays.shape[0], n,
                                          n, channels).astype('float32')
        image_data = (image_data*255).astype('uint8')
        image_labels = np.array([image_list[i][:][2]
                                 for i in range(len(image_list))])
        image_labels = keras.utils.to_categorical(image_labels, num_classes=2)

        test_set = data.flow(x=image_data, y=image_labels,
                             batch_size=batch_size,
                             shuffle=False)

    else:
        test_set = data.flow_from_directory(path, target_size=target_size,
                                            classes=classes,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            color_mode=color_mode)
    return test_set
