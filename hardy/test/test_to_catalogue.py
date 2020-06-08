import keras
import unittest

import numpy as np
import pandas as pd

from hardy.handling.to_catalogue import learning_set, test_set
from hardy.handling import to_catalogue as catalogue

path = './hardy/test/test_image/'
data_path = './hardy/test/test_data/'
split = 0.1
classes = ['class_1', 'class_2']
batch_size = 1


class TestSimulationTools(unittest.TestCase):

    def test_learning_set(self):
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

    def test_test_set(self):
        testing = test_set(path, batch_size=batch_size, iterator_mode=None,
                           classes=classes)
        assert isinstance(path, str), 'the path should be in a string format'
        assert isinstance(classes, list), \
            'the classes should be inputted as a list'
        for item in classes:
            assert isinstance(item, str), 'the class should be a string'
        assert isinstance(testing, keras.preprocessing.image.DirectoryIterator
                          ), 'the training set should be an image iterator'

    def test_save_load_data(self):
        # Simple pickeling save / load function
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
