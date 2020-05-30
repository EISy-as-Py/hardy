import keras
import unittest

import numpy as np

from hardy.classifier import Keras_CNN


# define variables to use for the following test:

path = './hardy/test/test_image/'
split = 0.1
batch_size = 1
classes = ['class_1', 'class_2']
epochs = 1
kernel_size = 3
activation = ['relu', 'relu', 'relu']
input_shape = (50, 50, 1)


class TestSimulationTools(unittest.TestCase):

    def test_learning_set(self):
        train, val = Keras_CNN.learning_set(path, split=split,
                                            batch_size=batch_size,
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
        testing = Keras_CNN.test_set(path, batch_size=batch_size,
                                     classes=classes)
        assert isinstance(path, str), 'the path should be in a string format'
        assert isinstance(classes, list), \
            'the classes should be inputted as a list'
        for item in classes:
            assert isinstance(item, str), 'the class should be a string'
        assert isinstance(testing, keras.preprocessing.image.DirectoryIterator
                          ), 'the training set should be an image iterator'

    def test_build_model(self):
        train, val = Keras_CNN.learning_set(path, split=split, classes=classes)
        model, history = Keras_CNN.build_model(train, val,
                                               epochs=epochs,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               input_shape=input_shape)
        assert isinstance(train, keras.preprocessing.image.DirectoryIterator),\
            'the training set should be an image iterator type of object'
        assert isinstance(val, keras.preprocessing.image.DirectoryIterator),\
            'the validation set should be an image iterator type of object'
        assert isinstance(kernel_size, int), \
            'the kernel size should beinputted as na integer'
        assert isinstance(activation, list), \
            'activation functions should be indicated using a list'
        for item in activation:
            assert isinstance(item, str), \
                'the activation function should be a string'
        assert isinstance(input_shape, tuple), \
            'the input shape should be indicated in a tuple'
        assert len(input_shape) == 3, 'the input shape should ahve 3 entries'
        assert input_shape[2] == (1 or 3), \
            'the image should be either grayscale or rgb' +\
            'wrong number of channels inputted'
        assert isinstance(model, keras.engine.sequential.Sequential),\
            'the CNN model should be a keras sequential model'
        assert isinstance(history, keras.callbacks.callbacks.History), \
            'the history should be the output of a allback function'

    def test_evaluate_model(self):
        # define the sets and the model to use for the rest of the testing
        train, val = Keras_CNN.learning_set(path, split=split, classes=classes)
        testing = Keras_CNN.test_set(path, batch_size=batch_size,
                                     classes=classes)
        model, history = Keras_CNN.build_model(train, val,
                                               epochs=epochs,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               input_shape=input_shape)
        results = Keras_CNN.evaluate_model(model, testing)
        assert isinstance(results, list), \
            'model performance should be store in a list'
        assert results[1] <= 1,\
            'the accuracy should be a number smaller than 1'

    def test_report_on_metrics(self):
        train, val = Keras_CNN.learning_set(path, split=split, classes=classes)
        testing = Keras_CNN.test_set(path, batch_size=batch_size,
                                     classes=classes)
        model, history = Keras_CNN.build_model(train, val,
                                               epochs=epochs,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               input_shape=input_shape)
        conf_matrix, report = Keras_CNN.report_on_metrics(
                                model, testing,
                                target_names=['noisy', 'not_noisy'])
        assert isinstance(conf_matrix, np.ndarray), \
            'the confusion matrix should be contained in a numpy array'
        assert isinstance(report, str), 'the report should be a string'
