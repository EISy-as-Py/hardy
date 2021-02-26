import keras
import pickle
import os
import time

# import matplotlib.pyplot as plt
import numpy as np

import hardy.handling.visualization as vis
import hardy.handling.handling as handling
from keras.preprocessing.image import ImageDataGenerator


"""
To_Catalogue, functions for handling (?) data?
    (This has gotten messy and VERY overlapping, both with handling.py
     and with the downstream pre_processing.py)...

Current Status:
    save_load_data : Executes a pickel data dump or load (unsafe?)

    _data_tuples_from_fnames :  Gets properly formatted List_of_tuples from
                                an input_path using smart category finder
                                and smart/safe read csv (arbitrary rows, and
                                deletes columns of "bad" data like strings)

    ** Note:  Here in Wrapping Function Flow, is where the Arbitrage Transforms
              Would "Intercept" the data and create transforms!

    rgb_list :  Takes that list of dataframe tuples (following convention)
                and creates a list of image tuples (same convention)

    data_set_split : Takes that image_tuple list, and splits
                     based on an also-given list of file names?
                     test_set is the names passed, learning is all others...

    rgb_visualize : Internal function called by rgb_list. Takes one fdata
                    dataframe and creates the image for it via a plot_format
                    string like 'RGBrgb' or "RgBrGb","Rg" or "Rxxgxx" etc.

    learning_set :  Complex (multi-options) BUT, given the right rules
                    (should be defaults) will turn the image_tuple_list
                    into the proper (255) images, then use the Keras
                    Preprocessing to turn them into keras objects.
                    Finally, executes data_set_split to return separate
                    Training and Validation data sets.

    test_set:       Does all of that, without splitting the data.

    #######################################################################
    rgb_list_to_dirflow: Unused for now. Gives the possibility of outputting
                         the rgb_list of tuples to a data structure
                         designed to work with keras flow_from_dir()
    _safe_clear_dirflow: Used to "safely" clear a directory to overwrite with
                         new files. Clears NOTHING unless specific folder
                         structure is in place, to avoid errors.

"""


def save_load_data(filename, data=None, save=None, load=None,
                   file_extension='.npy', location='./'):
    """Function to save and load data

    Function that can save or load data depending on given parameters.

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
        return 'Successfully Pickled'
    elif load:
        loaded_data = pickle.load(open(location + filename + file_extension,
                                  'rb'))
        return loaded_data


def _data_tuples_from_fnames(input_path='./', skiprows=0, classes=None):
    """
    Setting up the Data_tuples list, from ONE FOLDER with all of the data
    (OF different classes) inside of it.
    For each file, do a "smart-load" of the data, remove bad columns,
    and determine the classification from the file name.
    Then Return that line of data_tuples in the format of:
    (FileName (no extension), DataFrame, LABEL)
    """
    # Get list of classes for later
    list_of_tuples = []
    if classes is None:
        # This tells us to find the categories on our own.
        #    See "Handling" package for these methods.
        classes = handling.classes_from_fnames(os.listdir(input_path))
    elif type(classes) is list:
        # OR simply pass an integer of how many to expect.
        # (in the instance above, default is to expect 2)
        classes = handling.classes_from_fnames(os.listdir(input_path),
                                               expect=len(classes))
    else:
        pass

    # Now loop through each item in list, load the dataframe, and append
    # the SERIAL, Dataframe (fixed), and Class to the list of tuples!
    fread_timer = time.perf_counter()
    n_total = len(os.listdir(input_path))
    n_trigger = int(n_total/10)
    n_counter = 0
    n_reset = 0

    last_skiprows = None
    for entry in os.listdir(input_path):
        n_counter += 1
        n_reset += 1
        if n_reset >= n_trigger:
            fread_rate = int(n_trigger / (time.perf_counter() - fread_timer))
            if fread_rate == 0:
                fread_rate += 1
            # Rate in Files per Second.
            print('\rLoaded\t{} of {}\tFiles'.format(n_counter, n_total) +
                  '\t at rate of {} Files per Second'.format(fread_rate),
                  end='')
            fread_timer = time.perf_counter()
            n_reset = 0
        else:
            pass

        if entry.endswith('.csv'):
            # Read data into pandas dataframe
            fdata, last_skiprows = \
                handling.read_csv(input_path+entry,
                                  skiprows=skiprows,
                                  last_skiprows=last_skiprows)
            # Now remove any columns with bad data types
            # (Strings, objects, etc)
            for column in fdata.columns:
                if fdata[column].dtypes is float:
                    pass
                elif fdata[column].dtypes is np.dtype('float64'):
                    pass
                elif fdata[column].dtypes is int:
                    pass
                else:
                    # If type is not int, float, or numpy special float...
                    # It's either string, object, or something else bad..
                    fdata = fdata.drop(columns=column)

            label = None
            for each_label in classes:
                # Find the first label that matches.
                if not label and each_label in entry:
                    label = each_label
                else:
                    pass
            if not label:
                # If none of the labels fit, make new "not" first label
                label = "not_" + classes[0]
            else:
                pass

            list_of_tuples.append((entry.rstrip(entry[-4:]),
                                  fdata, label))
        else:
            # If File is not csv, ignore
            pass
    t_mins = round(n_total/fread_rate/60, 2)
    print("\n\t Success!\t About {} Minutes...".format(t_mins))
    # (Because timer has no Newline Character!)
    return list_of_tuples


def rgb_list(data_tuples, plot_format='RgBrGb', column_names=None,
             combine_method='add', scale=1.0, storage_location='./'):
    '''
    Input a path of csv files (with some guidance),
    Plot them RGB-wise into images
    return a list of tuples as to be fed into the keras PreProcess f(n)

    Parameters
    ----------
    data_tuples :   list of tuples
                    following the convention:
                    (SERIAL, DataFrame, LABEL)
                    (see below...)
    plot_format :   string
                    to pass into rgb_visualize
                        "single", "else", or some "RGBrgb"...
                        DEFAULT: "RgBrGb"? Discuss with group!!!
    combine_method :string
                    to pass into rgb_visualize

    column names :  list of strings (Optional)
                    IF given, will drop all columns not in the
                        list given. (If no colums match, will ERROR.)
    scale :  float
             percentage fo the image to reduce its size to.
    Returns
    -------
    list_of_rgb_tuples  :   list of tuples
                            following the format: (SERIAL, IMG, LABEL)
    SERIAL      :   File name with the extension taken off
                        (We should parse with . not just [-4])...
    IMG         :   ndarray of NxNx3

    LABEL       :   Classification label, either from the passed list
                        or from the last part of the serial/filename:
                        "123847_afsukjeh_*LABEL*.csv""

    '''
    pickle_file = open(storage_location, 'wb')
    print("Making rgb Images from Data...", end='\t')
    t = time.perf_counter()
    list_of_rgb_tuples = []
    for data_tuple in data_tuples:
        # For each dataframe given
        fdata = data_tuple[1]

        rgb_image = rgb_visualize(fdata, plot_format, combine_method,
                                  column_names, scale=scale)
        # Need some check that the visualization worked?

        rgb_tuple = (data_tuple[0], rgb_image, data_tuple[2])
        list_of_rgb_tuples.append(rgb_tuple)

    pickle.dump(list_of_rgb_tuples, pickle_file)
    pickle_file.close()
    t_sec = round(time.perf_counter()-t, 2)
    print("Success in {}seconds!".format(t_sec))
    return 0


def regular_plot_list(data_tuples, scale=1.0, storage_location='./'):
    '''
    Returns a list of tuples containing the arrays of images
    representing x-y plot

    Parameters
    ----------
    data_tuples :   list of tuples
                    The list of tuples in the following format
                     (filenames, dataframe, label)
    scale :  float
          percentage fo the image to reduce its size to.
    Returns
    -------
    list_of_plot_tuples  :  list of tuples
                            The list of tuples in the following format
                            (filename, image array, label)
    '''

    pickle_file = open(storage_location, 'wb')
    print("Making regular plot Images from Data...", end='\t')
    t = time.perf_counter()
    list_of_plot_tuples = []
    for data_tuple in data_tuples:
        # For each dataframe given
        fdata = data_tuple[1]

        plot_image = vis.regular_plot(fdata, scale=scale)
        # Need some check that the visualization worked?

        plot_tuple = (data_tuple[0], plot_image, data_tuple[2])
        list_of_plot_tuples.append(plot_tuple)

    pickle.dump(list_of_plot_tuples, pickle_file)
    pickle_file.close()

    t_sec = round(time.perf_counter()-t, 2)
    print("Success in {}seconds!".format(t_sec))
    return 0


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
                  column_names=None, scale=1.0):
    '''
    Input a list of dataframes (already read and/or processed),
    Plot them RGB-wise into images
    return a list of tuples as to be fed into the keras PreProcess f(n)

    Parameters
    ----------
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
    scale :  float
             percentage fo the image to reduce its size to.

    Returns
    -------

    list_of_tuples: list of tuples, following: (SERIAL, IMG, LABEL)

    SERIAL: File name with the extension taken off
            (We should parse with . not just [-4])

    IMG: ndarray of NxNx3

    '''
    if not column_names:
        column_names = list(fdata.columns)
    # This plot_format [single / else] could be removed. Discuss it
    if plot_format == 'single':
        rgb_image = vis.rgb_plot(red_array=fdata[column_names[0]],
                                 blue_array=fdata[column_names[1]],
                                 plot=False, scale=scale)
    elif plot_format == "else":
        rgb_image_x = vis.rgb_plot(red_array=fdata[column_names[0]],
                                   plot=False, scale=scale)
        rgb_image_y = vis.rgb_plot(blue_array=fdata[column_names[1]],
                                   plot=False, scale=scale)
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
            if R is None and plot_format[i] == "R":
                R = fdata[column_names[i]]
            if G is None and plot_format[i] == "G":
                G = fdata[column_names[i]]
            if B is None and plot_format[i] == "B":
                B = fdata[column_names[i]]
            if r is None and plot_format[i] == "r":
                r = fdata[column_names[i]]
            if g is None and plot_format[i] == "g":
                g = fdata[column_names[i]]
            if b is None and plot_format[i] == "b":
                b = fdata[column_names[i]]
        rgb_image_x = vis.rgb_plot(red_array=R, green_array=G,
                                   blue_array=B, plot=False, scale=scale)
        rgb_image_y = vis.rgb_plot(red_array=r, green_array=g,
                                   blue_array=b, plot=False, scale=scale)

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

# The following funciton needs to be revised and connected
#  to the amin code wrapping function
# def rgb_list_to_DirFlow(rgb_tuples, basepath, newfolder="rbg_for_keras",
#                         delete_existing=True):
#     """
#     Takes the list of tuples (as made in "rgb_list") and creates the exact
#         file structure of saved PNG images that will be used in the
#         "KERAS FLOW FROM DIRECTORY" method.
#
#     Also will save a log file in the base path (csv? or look for log?)
#         describing the
#     """
#     classes = []
#     for each_image in rgb_tuples:
#         if each_image[2] not in classes:
#             # Make a unique list of the classes...
#             classes.append(each_image[2])
#         else:
#             pass
#
#     newfolder_path = os.path.join(basepath, newfolder)
#     # Make new folder full path... But what if it exists?
#     if os.path.isdir(newfolder_path):
#         # Well, if "Delete Existing" is true, delete that folder.
#         if delete_existing:
#             # Loop through and delete all... DANGEROUS
#             # So instead wrote a safety loop.
#             _safe_clear_dirflow(newfolder_path)
#         else:
#             n = 0
#            newer_folder = os.path.join(basepath, newfolder + "_{}".format(n))
#             while os.path.isdir(newer_folder) and n < 100:
#                 n += 1
#                 newer_folder = os.path.join(basepath,
#                                             newfolder + "_{}".format(n))
#             else:
#                 pass
#             newfolder_path = newer_folder
#             print("Directory Full! \n" +
#                   "Creating New Dir @   {}".format(newer_folder))
#     else:
#         pass
#
#     # Now, make folders and fill them with the images!
#     os.makedirs(newfolder_path)
#     for each_class in classes:
#         class_folder = os.path.join(newfolder_path, each_class)
#         os.makedirs(class_folder)
#         for each_image in rgb_tuples:
#             if each_image[2] == each_class:
#                 # If this image is in the class, save it in this folder!
#                 save_png = os.path.join(class_folder, each_image[0] + '.png')
#                 plt.imsave(save_png, each_image[1])
#     return basepath, newfolder


def _safe_clear_dirflow(the_path):
    """
    Safely check that the path contains ONLY folders of png files.
        if any other structure, will simply ERROR out.
    (for now, doesn't fix errors, just raises them)
    """
    print("Clearing {}...".format(the_path))
    assert os.path.isdir(the_path), "Didn't pass a folder to be cleaned"
    list_dir = [f for f in os.listdir(the_path) if not f.startswith('.')]
    for folder in list_dir:
        cat_folder = os.path.join(the_path, folder)
        assert os.path.isdir(cat_folder), \
            "Dir contains Non-Folder File!"
        cat_folder_item = [f for f in os.listdir(cat_folder)
                           if not f.startswith('.')]
        for file in cat_folder_item:
            # For every file, confirm is PNG or error.
            # DONT DELETE YET, IN CASE OF ERRORS!
            assert ".png" in file, "Folder has Non PNG Contents!"
    # If we got though that with no error, then now we can delete!
    # for folder in os.listdir(the_path):
    #     cat_folder = os.path.join(the_path, folder)
    #     for file in os.listdir(cat_folder):
    #         os.remove(os.path.join(cat_folder, file))
    #     os.rmdir(cat_folder)
    # os.rmdir(the_path)
    return True


def learning_set(path=None, split=0.1, target_size=(80, 80),
                 classes=['noisy', 'not_noisy'], batch_size=32,
                 color_mode='rgb', iterator_mode='arrays',
                 image_list=None, k_fold=None, k=None, fold=None, **kwargs):
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
    iterator_mode: str
                    string indicating which Keras IamgeDataGenerator mode
                    to use. Options are 'arrays' or 'images'. The first will
                    use the "flow" option, the second will use
                    "flow_from_directory" option
    image_list: list
                 The list of tuples in the following format
                 (filenames, image_array, label)
    k_fold: Bool
    k:  int
    fold: int

    Returns
    -------
    training_set:  Keras image iterator
                The training set containg labelled images
    validation_set: Keras image iterator
                The training set containg labelled images
    '''

    if iterator_mode == 'arrays':
        n = target_size[0]
        if color_mode == 'rgb':
            channels = 3
        else:
            channels = 1

        assert image_list, 'the image arrays should be provided'
        # To Do :Add checks for the image arrays- (filename, arrays, label)

        image_arrays = np.array([image_list[i][1][:]
                                for i in range(len(image_list))])

        image_data = image_arrays.reshape(image_arrays.shape[0], n,
                                          n, channels).astype('float32')
        image_data = (image_data*255).astype('uint8')
        image_labels = np.array([image_list[i][:][2]
                                 for i in range(len(image_list))])
        for i, label in enumerate(np.unique(image_labels)):
            for j in range(len(image_labels)):
                if image_labels[j] == label:
                    image_labels[j] = i

        if len(np.unique(image_labels)) != len(np.unique(classes)):
            print('The number of unique labels was found to be {},'
                  ' expected {}'.format(len(np.unique(image_labels)),
                                        len(np.unique(classes))))

        if k_fold:

            assert k, 'The number of folds needs to be provided'

            image_data_list = [(image_data[i], image_labels[i])
                               for i in range(len(image_data))]
            np.random.shuffle(image_data_list)

            num_validation_samples = len(image_data_list) // k
            # define the training and validation set for the given fold

            x_train = np.array(
                [image_data_list[i][0] for i in np.concatenate(
                    (np.arange(0, num_validation_samples*fold),
                     np.arange(num_validation_samples*(fold+1),
                               len(image_data_list))))])

            y_train = [image_data_list[i][1] for i in
                       np.concatenate((
                           np.arange(0, num_validation_samples*fold),
                           np.arange(num_validation_samples*(fold+1),
                                     len(image_data_list))))]
            y_train = keras.utils.to_categorical(
                y_train, num_classes=len(np.unique(image_labels)))

            x_val = np.array(
                [image_data_list[i][0] for i in np.arange(
                    num_validation_samples*fold,
                    num_validation_samples*(fold+1))])

            y_val = [image_data_list[i][1] for i in
                     np.arange(num_validation_samples*fold,
                     num_validation_samples*(fold+1))]
            y_val = keras.utils.to_categorical(
                y_val, num_classes=len(np.unique(image_labels)))

            data = ImageDataGenerator(**kwargs)

            training_set = data.flow(x=x_train, y=y_train,
                                     shuffle=False,
                                     batch_size=batch_size)
            validation_set = data.flow(x=x_val, y=y_val,
                                       shuffle=False,
                                       batch_size=batch_size)
        else:
            image_labels = keras.utils.to_categorical(
                image_labels, num_classes=len(np.unique(image_labels)))

            if split == 0:
                data = ImageDataGenerator(**kwargs)

                training_set = data.flow(x=image_data, y=image_labels,
                                         batch_size=batch_size, shuffle=False)
                validation_set = []
            else:
                data = ImageDataGenerator(validation_split=split, **kwargs)

                training_set = data.flow(x=image_data, y=image_labels,
                                         batch_size=batch_size,
                                         subset='training')
                validation_set = data.flow(x=image_data, y=image_labels,
                                           batch_size=batch_size,
                                           subset='validation')

    else:
        data = ImageDataGenerator(validation_split=split, **kwargs)
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


def test_set(path=None, target_size=(80, 80),
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
    iterator_mode : str
                    string indicating which Keras IamgeDataGenerator mode
                    to use. Options are 'arrays' or 'images'. The first will
                    use the "flow" option, the second will use
                    "flow_from_directory" option
    image_list : list
                 The list of tuples in the following format
                 (filenames, image_array, label)

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
        # To Do : Add checks for the image arrays- (filename, arrays, label)

        image_arrays = np.array([image_list[i][1][:]
                                for i in range(len(image_list))])

        if len(image_arrays[0][1]) != target_size[0]:
            print('The expected target size is {}, found {}'
                  .format(len(image_arrays[0][1]), target_size[0]))
            n = len(image_arrays[0][1])

        image_data = image_arrays.reshape(image_arrays.shape[0], n,
                                          n, channels).astype('float32')
        image_data = (image_data*255).astype('uint8')
        image_labels = np.array([image_list[i][:][2]
                                 for i in range(len(image_list))])
        for i, label in enumerate(np.unique(image_labels)):
            for j in range(len(image_labels)):
                if image_labels[j] == label:
                    image_labels[j] = i
        if len(np.unique(image_labels)) != len(np.unique(classes)):
            print('The number of unique labels was found to be {},'
                  ' expected {}'.format(len(np.unique(image_labels)),
                                        len(np.unique(classes))))
        image_labels = keras.utils.to_categorical(
            image_labels, num_classes=len(np.unique(image_labels)))
        test_set = data.flow(x=image_data, y=image_labels,
                             batch_size=batch_size,
                             shuffle=False)

    else:
        test_set = data.flow_from_directory(path + 'test_set/',
                                            target_size=target_size,
                                            classes=classes,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            color_mode=color_mode)
    return test_set
