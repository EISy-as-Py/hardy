# import keras
import unittest

import numpy as np
import pandas as pd

from hardy.handling.to_catalogue import learning_set, test_set
from hardy.handling import to_catalogue as catalogue
from hardy.handling import pre_processing as preprocessing

path = './hardy/test/test_image/'
data_path = './hardy/test/test_data/'


class TestSimulationTools(unittest.TestCase):

    def test_hold_out_test_set(self):
        num_files = 5
        # Frm .csv files
        test_set_filenames = preprocessing.hold_out_test_set(
            data_path, number_of_files_per_class=num_files)
        assert isinstance(test_set_filenames, list), 'format should be a list'
        assert len(test_set_filenames) == 2*num_files, \
            'the test set is not the correct length'
        assert len(np.unique(test_set_filenames)) == 2*num_files,\
            'the files selected to compose the test set should be unique'
        # from tuples
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)

        plot_tups = catalogue.rgb_list(data_tups)

        test_set_filenames = preprocessing.hold_out_test_set(
            image_list=plot_tups, number_of_files_per_class=num_files,
            iterator_mode='arrays')
        assert isinstance(test_set_filenames, list), 'format should be a list'
        assert len(test_set_filenames) == 2*num_files, \
            'the test set is not the correct length'
        pass

    def test_set_folder(self):
        
        pass

    def test_classes_folder_split(self):
        pass

    def test_save_to_folder(self):
        pass
