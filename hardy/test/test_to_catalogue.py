import keras
import unittest

import numpy as np
import pandas as pd

from hardy.handling.to_catalogue import learning_set, test_set
from hardy.handling import to_catalogue as catalogue
from hardy.handling import pre_processing as preprocessing

path = './hardy/test/test_image/'
data_path = './hardy/test/test_data/'
split = 0.1
classes = ['class_1', 'class_2']
batch_size = 1


class TestSimulationTools(unittest.TestCase):

    def test_learning_set(self):
        # Directory iterator option
        train, val = learning_set(path, split=split,
                                  batch_size=batch_size,
                                  iterator_mode=None,
                                  classes=classes)
        assert isinstance(path, str), 'the path should be in a string format'
        assert isinstance(split, (float, np.float32, int)), \
            ' the data split should be a number'
        assert isinstance(classes, list), \
            'the classes should be inputted as a list'
        for item in classes:
            assert isinstance(item, str), 'the class should be a string'
        assert isinstance(train, keras.preprocessing.image.DirectoryIterator),\
            'the training set should be an image iterator type of object'
        assert isinstance(val, keras.preprocessing.image.DirectoryIterator),\
            'the validation set should be an image iterator type of object'
        assert isinstance(batch_size, int), \
            'the batch size should be an integer'
        # array iterator option
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)
        plot_tups = catalogue.rgb_list(data_tups)
        train, val = learning_set(image_list=plot_tups, split=split,
                                  batch_size=batch_size,
                                  iterator_mode='arrays',
                                  classes=classes)
        assert isinstance(train, keras.preprocessing.image.NumpyArrayIterator
                          ), \
            'the training set should be an image iterator type of object'
        assert isinstance(val, keras.preprocessing.image.NumpyArrayIterator),\
            'the validation set should be an image iterator type of object'

    def test_test_set(self):
        # Directory iterator option
        testing = test_set(path, batch_size=batch_size, iterator_mode=None,
                           classes=classes)
        assert isinstance(path, str), 'the path should be in a string format'
        assert isinstance(classes, list), \
            'the classes should be inputted as a list'
        for item in classes:
            assert isinstance(item, str), 'the class should be a string'
        assert isinstance(testing, keras.preprocessing.image.DirectoryIterator
                          ), 'the training set should be an image iterator'
        # array iterator option
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)
        plot_tups = catalogue.rgb_list(data_tups)
        testing = test_set(image_list=plot_tups, batch_size=batch_size,
                           iterator_mode='arrays', classes=classes)
        assert isinstance(testing, keras.preprocessing.image.NumpyArrayIterator
                          ), 'the training set should be an image iterator'

    def test_save_load_data(self):
        # Simple pickeling save / load function
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)
        plot_tups = catalogue.rgb_list(data_tups)
        data = catalogue.save_load_data('test_pickled_data', data=plot_tups,
                                        save=True, location='./hardy/test/')
        assert data == 'Successfully Pickled'
        data = catalogue.save_load_data('test_pickled_data', data=plot_tups,
                                        load=True, location='./hardy/test/')
        assert isinstance(data, list), 'the data was correctly loaded'
        pass

    def test_data_tuples_from_fnames(self):
        """
        Testing Fn for the List-Of-Tuples function Wrapper
        Largest wrapper of this file set.
        (Given just the folder with CSV files in it, will generate the
             designated "List-Of-Tuples" of raw data).
        """
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path,
                                                       skiprows=6,
                                                       classes=None)
        assert type(data_tups) is list,\
            "Data List-of-Tuples did not return a List"

        for row in data_tups:
            assert type(row) is tuple, "List-of-Tuples has non-tuple?"
            assert type(row[0]) is str, "File Name in Tuple is wrong format."
            assert type(row[1]) is pd.DataFrame,\
                "List-of-Tuples improperly importing data"
            assert type(row[2]) is str, "Class label is not a string?"
        pass

    def test_rgb_list(self):
        """
        Testing the Tuple-List image visualization wrapper
            Inputs the raw tuple-list from the prior wrapper
            and performs the visualizations as called in the "standard"
            methods.
        """

        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)

        plot_tups = catalogue.rgb_list(data_tups)

        for row in plot_tups:
            assert type(row) is tuple, "List-of-Tuples has non-tuple?"
            assert type(row[0]) is str, "File Name in Tuple is wrong format."
            assert type(row[1]) is np.ndarray,\
                "List-of-image-Tuples is not in np.ndarray format??"
            assert type(row[2]) is str, "Class label is not a string?"
        pass

    def test_rgb_visualize(self):
        """
        Individual data frame image maker. This is included in prior wrapps
            so doesn't really need much testing? but not sure how coveralls
            works so I guess we'll do it.
        """
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)

        fdata = data_tups[0][1]
        assert type(fdata) is pd.DataFrame, "Need testing dataframe"

        image_arr = catalogue.rgb_visualize(fdata)

        assert image_arr.shape[2] == 3,\
            "Expected NxNx3 Image. Instead got {}".format(image_arr.shape)
#

    def test_data_split(self):
        num_files = 3
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)

        plot_tups = catalogue.rgb_list(data_tups)

        test_set_filenames = preprocessing.hold_out_test_set(
            data_path, number_of_files_per_class=num_files)
        test_set_list, learning_set_list = catalogue.data_set_split(
            plot_tups, test_set_filenames)
        assert isinstance(test_set_filenames, list), 'format should be a list'
        assert len(test_set_filenames) == 2*num_files, \
            'the test set is not the correct length'
        assert isinstance(test_set_list, list), 'format should be a list'
        assert isinstance(learning_set_list, list), 'format should be a list'
        assert len(test_set_list) == 2*num_files, \
            'the test_set_list is not the correct length'
        pass
