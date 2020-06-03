import keras
import unittest

import numpy as np

from hardy.handling.to_catalogue import learning_set, test_set

path = './hardy/test/test_image/'
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
