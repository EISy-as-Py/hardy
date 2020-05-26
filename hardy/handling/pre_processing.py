import os
import random
import shutil


def hold_out_test_set(path, number_of_files_per_class, classes=['noise', ''],
                      file_extension='.csv'):
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

    Returns
    -------
    test_set_serialnumbers : list
                             A list containig the strings of filenames
                             randomly selected to be part of the test set.
    '''

    # Initialize a list that will contain the serial numbers of thefiles
    # composing the test set
    test_set_serialnumbers = []

    # Randomly pick files that are labelled as noisy and append
    #  them into the test_set list
    file_list_1 = [n for n in os.listdir(path)
                   if n.endswith(classes[0]+file_extension)]
    file_list_2 = [n for n in os.listdir(path)
                   if not n.endswith(classes[0]+file_extension)]
    for i in range(number_of_files_per_class):
        chosen_file = random.choice(file_list_1)
        file_list_1.remove(chosen_file)
        test_set_serialnumbers.append(str(chosen_file).split('_')[0])

        chosen_file = random.choice(file_list_2)
        file_list_2.remove(chosen_file)
        test_set_serialnumbers.append(str(chosen_file).split('_')[0])

    return test_set_serialnumbers


def test_set_folder(path, test_set_serialnumbers):
    '''
    Functions that removes the files randomly chosen to be part of the
    test set and saves them intothe test_set folder

    Parameters
    ----------
    path : str
           string containing the path where to create a test set folder
    test_set_serialnumbers: list
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

    for serial_number in test_set_serialnumbers:
        test_set_files = [n for n in os.listdir(path)
                          if n.startswith(serial_number)]
        for file in test_set_files:
            shutil.move(path + file, test_set_folder)
    return test_set_folder


def classes_folder_split(path, plot_type=None, classes=['noise', ''],
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
    plot_type: str
               a string containig indication of wht plot type the file
               represent. The plot type is contained in the filename as a
               label at the end.
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
    if plot_type:
        end_of_file = '_'+plot_type + file_extension
    else:
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
