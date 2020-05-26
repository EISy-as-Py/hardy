import os
import random
import shutil
import sys

import numpy as np
import pandas as pd


def hold_out_test_set(path, number_of_files_per_class, classes=['noise', ''],
                      file_extension='.csv'):
    '''
    Functions that returns a list of filenames
    of the randomly selected files to compose the test set

    Parameters
    ----------

    Returns
    -------
    '''
    test_set_filenames = []

    file_list_1 = [n for n in os.listdir(path) if
                   n.endswith(classes[0] + file_extension)]
    file_list_2 = [n for n in os.listdir(path) if not
                   n.endswith(classes[0] + file_extension)]
    for i in range(number_of_files_per_class):
        chosen_file = random.choice(file_list_1)
        file_list_1.remove(chosen_file)
        test_set_filenames.append(str(chosen_file).rstrip(
                                  random.choice(list_of_files_0)[-4:])+'.png')
        chosen_file = random.choice(file_list_2)
        file_list_1.remove(chosen_file)
        test_set_filenames.append(str(chosen_file).rstrip(
                                  random.choice(list_of_files_1)[-4:])+'.png')
    return test_set_filenames


def test_set_folder(path, test_set_filenames):
    '''
    Functions that removes the files randomly chosen to be part of the
    test set and saves them intothe test_set folder

    Parameters
    ----------

    Returns
    -------
    '''

    test_set_folder = path + 'test_set/'

    if not os.path.exists(test_set_folder):
        os.makedirs(test_set_folder)

    for file in [n for n in os.listdir(path) if n in test_set_filenames]:
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

    Returns
    -------
    '''
    assert len(classes) == len(class_folder), 'the number of labels and' +\
        'folders created needs to be equal'
    list_of_folders = []
    for i in range(len(classes)):
        list_of_files = [n for n in os.listdir(path) if
                         n.endswith(classes[i] + file_extension)]
        new_folder_path = path + class_folder[i] + '/'
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        for file in list_of_files:
            shutil.move(path + file, new_folder_path)
        list_of_folders.append(new_folder_path)
    return
