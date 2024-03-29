import os
import random
import shutil


def hold_out_test_set(path=None, number_of_files_per_class=100, seed=None,
                      classes=['noise', ''], file_extension='.csv'):
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
    classes.sort(reverse=True)

    test_set_filenames = []

    whole_list = os.listdir(path)

    if seed:
        random.seed(seed)

    for i in range(len(classes)):

        file_list = [n for n in whole_list
                     if n.endswith(classes[i]+file_extension)]

        whole_list = [item_i for item_i in whole_list
                      if item_i not in file_list]

        file_list_for_selection = file_list

        for i in range(number_of_files_per_class):
            chosen_file = random.choice(file_list_for_selection)
            file_list_for_selection.remove(chosen_file)
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


def save_to_folder(input_path, project_name, run_name):
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
    run_name : str
               String representing the transformation applied to
               the data

    Returns
    -------
    transformation_folder_path :  str
                                  String representing the path to the newly
                                  generated path
    '''

    hardy_folder_path = input_path + project_name + '/'
    if not os.path.exists(hardy_folder_path):
        try:
            os.makedirs(hardy_folder_path)
        except OSError:
            pass

    transformation_folder_path = hardy_folder_path + run_name + '/'

    if not os.path.exists(transformation_folder_path):
        os.makedirs(transformation_folder_path)

    return transformation_folder_path
