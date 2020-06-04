import os
import random
import shutil
import time
"""
NOTE: CURRENTLY OVERLAPPING WITH TO_CATALOGUE.
      GOAL IS TO SEPARATE EVENTUALLY BASED ON DEPENDANCY.

Current Functionality:
    Given raw data Path,

Current WIP:
    I think this should contain the Wrappers around arbitrage and the
        functions that are in to_catalogue.py,
        so that this will have the "One Click" function to make
        List_of_Tuples.
    SPLITTING RULES + Path --? ?
    path --> list of files (and classes?)
    files --> list_of_tuples (RAW)
    raw --> list_of_tuples (Transformed)
    transformed --> list_of_tuples (RGB images)
    l_o_t(rgb) --> "keras preprocessed" data set.

"""
import hardy.handling.to_catalogue as catalogue
import hardy.arbitrage.arbitrage as arbitrage


import keras_ready_frompath(raw_datapath,
                            tform_commands=None)



def hold_out_test_set(path=None, number_of_files_per_class=100,
                      classes=['noise', ''], file_extension='.csv',
                      image_list=None, iterator_mode=None):
    '''
    Functions that returns a list of filenames
    of the randomly selected files to compose the test set

    Parameters
    ----------
    path : str
           string containing the path to the files to select from
           the test set from.

    number_of_files_per_class: int
                               The number of files to select from each class.

    classes: list
             a list containing strings of the classes the data is divided in.
             The classes are contained in the filename as labels.

    file_extension: str
                    the extension of the file to read. The default value is
                    .csv
    image_list: np.array
                numpy array representing file names, image data and labels
    iterator_mode: str
                   string representing if the data provided is in arrays

    Returns
    -------
    test_set_serialnumbers : list
                             A list containig the strings of filenames
                             randomly selected to be part of the test set.
    '''

    # Initialize a list that will contain the serial numbers of thefiles
    # composing the test set
    test_set_filenames = []

    # seperating test_set_filenames for input as arrays
    if iterator_mode == "arrays":
        file_list_1 = [n[0] for n in image_list
                       if n[0].endswith(classes[0])]
        file_list_2 = [n[0] for n in image_list
                       if not n[0].endswith(classes[0])]
        for i in range(number_of_files_per_class):
            chosen_file = random.choice(file_list_1)
            file_list_1.remove(chosen_file)
            test_set_filenames.append(str(chosen_file))
            chosen_file = random.choice(file_list_2)
            file_list_2.remove(chosen_file)
            test_set_filenames.append(str(chosen_file))
    # # These lines are hardcoded to allow for 2 classes only # #
    # #  Rewrite to support a higher number of classes # #

    # Randomly pick files that are labelled as noisy and append
    #  them into the test_set list
    else:
        file_list_1 = [n for n in os.listdir(path)
                       if n.endswith(classes[0]+file_extension)]
        file_list_2 = [n for n in os.listdir(path)
                       if not n.endswith(classes[0]+file_extension)]
        for i in range(number_of_files_per_class):
            chosen_file = random.choice(file_list_1)
            file_list_1.remove(chosen_file)
            test_set_filenames.append(str(chosen_file.rstrip(
                                          chosen_file[-4:])))

            chosen_file = random.choice(file_list_2)
            file_list_2.remove(chosen_file)
            test_set_filenames.append(str(chosen_file.rstrip(
                                          chosen_file[-4:])))

    return test_set_filenames


def test_set_folder(path, test_set_filenames):
    '''
    Functions that removes the files randomly chosen to be part of the
    test set and saves them intothe test_set folder

    Parameters
    ----------
    path : str
           string containing the path where to create a test set folder
    test_set_filenames: list
                            The list containig the strings of filenames
                            randomly selected to be part of the test set.

    Returns
    -------
    test_set_folder : str
                      A string containging the path to the test set folder.
    '''

    test_set_folder = path + 'test_set/'

    if not os.path.exists(test_set_folder):
        os.makedirs(test_set_folder)

    test_set_files = [n for n in os.listdir(path) if n in test_set_filenames]

    for file in test_set_files:
        shutil.move(path + file, test_set_folder)
    return test_set_folder


def classes_folder_split(path, classes=['noise', ''],
                         class_folder=['noisy', 'not_noisy'],
                         file_extension='.png'):
    '''
    Functions that separates the files into folders
    representing each class

    Parameters
    ----------
    path : str
           string containing the path to the files where to create the
           training and validation sets folders
    classes: list
             A list containing strings of the classes the data is divided in.
             The classes are contained in the filename as labels.
    class_folder: list
                  A list of string containing the name of the folders to be
                  create to split the files into the right classes.
    file_extension: str
                    the extension of the file to be moved. The default value is
                    .png

    Returns
    -------
    list_of_folders: list
                     A list of stings representing the path of the new folders
                     created while splitting the data into classes
    '''
    assert len(classes) == len(class_folder), 'the number of labels and' +\
        'folders created needs to be equal'
    list_of_folders = []
    end_of_file = file_extension
    for i in range(len(classes)):
        list_of_files = [n for n in os.listdir(path) if
                         n.endswith(classes[i] + end_of_file)]
        new_folder_path = path + class_folder[i] + '/'
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        for file in list_of_files:
            shutil.move(path + file, new_folder_path)
        list_of_folders.append(new_folder_path)
    return list_of_folders


def save_to_folder(input_path, project_name, transformation_name):
    '''
    Function that creates a new path to the folder for a specific
    transformation. The transformation folder will be nested in a
    run folder named using the run date and the project name

    Parameters
    ----------
    input_path : str
                 String containing the path to the .csv files
    project_name : str
                   String representing the project name. This will be used to
                   name the folder containing the results from the hardy run
    transformation_name : str
                          String representing the transformation applied to
                          the data

    Returns
    -------
    transformation_folder_path :  str
                                  String representing the path to the newly
                                  generated path
    '''
    date = time.strftime('%y%m%d', time.localtime())

    hardy_folder_path = input_path + date + '/' + project_name + '/'
    if not os.path.exists(hardy_folder_path):
        os.makedirs(hardy_folder_path)

    transformation_folder_path = hardy_folder_path + transformation_name + '/'

    if not os.path.exists(transformation_folder_path):
        os.makedirs(transformation_folder_path)

    return transformation_folder_path
