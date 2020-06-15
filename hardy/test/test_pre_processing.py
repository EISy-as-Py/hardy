import os
import shutil
import unittest

import numpy as np
# import pandas as pd

# from hardy.handling.to_catalogue import learning_set, test_set
# from hardy.handling import to_catalogue as catalogue
from hardy.handling import pre_processing as preprocessing

data_path = './hardy/test/test_folder_split/'


class TestSimulationTools(unittest.TestCase):

    def test_hold_out_test_set(self):
        num_files = 2
        # Frm .csv files
        test_set_filenames = preprocessing.hold_out_test_set(
            data_path, number_of_files_per_class=num_files)
        assert isinstance(test_set_filenames, list), 'format should be a list'
        assert len(test_set_filenames) == 2*num_files, \
            'the test set is not the correct length'
        assert len(np.unique(test_set_filenames)) == 2*num_files,\
            'the files selected to compose the test set should be unique'
        # from tuples
        # data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)

        # plot_tups = catalogue.rgb_list(data_tups)

        # test_set_filenames = preprocessing.hold_out_test_set(
        #     image_list=plot_tups, number_of_files_per_class=num_files,
        #     iterator_mode='arrays')
        # assert isinstance(test_set_filenames, list),
        # 'format should be a list'
        # assert len(test_set_filenames) == 2*num_files, \
        #     'the test set is not the correct length'
        pass

    def test_set_folder(self):
        num_files = 5
        # Frm .csv files
        test_set_filenames = preprocessing.hold_out_test_set(
            data_path, number_of_files_per_class=num_files)
        test_folder = preprocessing.test_set_folder(
            data_path, test_set_filenames)
        assert isinstance(test_folder, str), \
            'the return should be the path to the test folder'
        assert os.path.exists(test_folder), \
            'the test folder was not correctly created'
        pass

    def test_classes_folder_split(self):
        list_of_folders = preprocessing.classes_folder_split(
            data_path, classes=['noise', ''],
            class_folder=['noisy', 'not_noisy'], file_extension='.csv')
        assert isinstance(list_of_folders, list), \
            'the output should be a list'
        for path in list_of_folders:
            assert os.path.exists(path), \
                'the class fodler was not correctly created'
            # move the files out of the folder for ensuring
            # the next test iteration works
            for file in [n for n in os.listdir(path)]:
                shutil.move(path + file, data_path)
        pass

    def test_save_to_folder(self):
        folder_path = preprocessing.save_to_folder(
            data_path, 'test_project', 'test_1')
        assert os.path.exists(folder_path), \
            'the fodler was not correctly created'
        pass
