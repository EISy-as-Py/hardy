import keras
import unittest
import pickle

import matplotlib.pyplot as plt
import numpy as np

from hardy.recognition import cnn
from hardy.handling.to_catalogue import learning_set, test_set

# define variables to use for the following test:

path = './hardy/test/test_image/'
split = 0.25
classes = ['class_1', 'class_2']
batch_size = 1


class TestSimulationTools(unittest.TestCase):

    def test_build_model(self):
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path='./hardy/recognition/')
        assert isinstance(train, keras.preprocessing.image.DirectoryIterator),\
            'the training set should be an image iterator type of object'
        assert isinstance(val, keras.preprocessing.image.DirectoryIterator),\
            'the validation set should be an image iterator type of object'
        assert isinstance(model, keras.engine.sequential.Sequential),\
            'the CNN model should be a keras sequential model'
        assert isinstance(history, keras.callbacks.callbacks.History), \
            'the history should be the output of a allback function'

    def test_evaluate_model(self):
        # define the sets and the model to use for the rest of the testing
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        testing = test_set(path, batch_size=batch_size, classes=classes,
                           iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path='./hardy/recognition/')
        results = cnn.evaluate_model(model, testing)
        assert isinstance(results, list), \
            'model performance should be store in a list'
        assert results[1] <= 1,\
            'the accuracy should be a number smaller than 1'

    def test_report_on_metrics(self):
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        testing = test_set(path, batch_size=batch_size, classes=classes,
                           iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path='./hardy/recognition/')
        conf_matrix, report = cnn.report_on_metrics(
                                model, testing,
                                target_names=['noisy', 'not_noisy'])
        assert isinstance(conf_matrix, np.ndarray), \
            'the confusion matrix should be contained in a numpy array'
        assert isinstance(report, str), 'the report should be a string'

    def test_feature_maps(self):
        with open('./hardy/test/test_model/test_model.sav', 'rb') as file:
            model = pickle.load(file)
        image_name = '200504-0132_sim_one_current_noise_log-freq.png'
        image_path = './hardy/test/test_image/class_1/'
        last_layer = cnn.feature_map(image_name, model, 2, 80,
                                     layer_num='last',
                                     image_path=image_path)
        assert isinstance(last_layer, np.ndarray), 'Invalid output generated'
        assert isinstance(last_layer[0][0], np.float32),\
            'The output returned is invalid'
        assert isinstance(last_layer[0][1], np.float32),\
            'The output returned is invalid'
        assert last_layer[0][0] or last_layer[0][1] > 0,\
            'The output returned is invalid'
        assert last_layer[0][0] or last_layer[0][1] < 1,\
            'The output returned is invalid'

    def test_plot_history(self):
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path='./hardy/recognition/')

        _, ax = plt.subplots(1, 2)
        ax = cnn.plot_history(history)
        epochs, loss = ax[0].lines[0].get_xydata().T
        assert (loss == history.history['loss']).all(), \
            'the plot should containg the loss value per epoch'
        epochs, acc = ax[1].lines[0].get_xydata().T
        assert (acc == history.history['accuracy']).all(), \
            'the plot should contain the accuracy value epoch'
        pass

    def test_save_load_model(self):
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path='./hardy/recognition/')

        saved_model = cnn.save_load_model(
            model=model, save=True, filepath='./hardy/test/model')
        assert saved_model == 'the model was correctly saved'
        model_loaded = cnn.save_load_model(
            load=True, filepath='./hardy/test/model')
        assert model_loaded, 'the model was not correctly loaded'
        pass

    def test_feature_map_layers(self):
        pass
