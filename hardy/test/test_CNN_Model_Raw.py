import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unittest

import hardy.classifier.CNN_Model_Raw as CNN

class TestCNNModelTools(unittest.TestCase):
    def test_DataImporter_Training(self):
        k = 1
        path_List_training = ['.', '.']
        image_width = 200
        image_height = 134
        train_d = CNN.EISDataImport.DataImporter_Training(self, k,
                                                          path_List_training,
                                                          image_width,
                                                          image_height)
        assert k == (len(path_List_training) - 1), 'Incorrect number of type'
        assert k <= 7, 'Too many types.'
        assert type(path_List_training) == list, \
            'path_List_training should be a list'
        assert image_width <= 1000, 'Image size is too large'
        assert image_height <= 1000, 'Image size is too large'

    def test_DataImporter_Predict(self):
        k = 1
        path_List_predict = ['.', '.']
        image_width = 200
        image_height = 134
        predict_d = CNN.EISDataImport.DataImporter_Predict(self, k,
                                                           path_List_predict,
                                                           image_width,
                                                           image_height)
        assert k == (len(path_List_predict) - 1), \
            'Incorrect number of folders/paths'
        assert k <= 10, 'Too many folders/paths.'
        assert type(path_List_predict) == list, \
            'path_List_predict should be a list'
        assert image_width <= 1000, 'Image size is too large.'
        assert image_height <= 1000, 'Image size is too large.'

def test_Build_Data(self):
    Training = True
    k = 1
    path_list = ['.', '.']
    image_width = 200
    image_height = 134
    Build_d = CNN.Build_Data(Training, k, path_list,
                             image_width, image_height)
    assert Training != Predict, 'Build only one type of data in one time.'
    if Training is True:
        assert k <= 7, 'Too many types.'
    if Predict is True:
        assert k <= 10, 'Too many folders/paths.'
    assert k == (len(path_list) - 1), 'Incorrect number of type.'
    assert type(path_list) == list, 'path_list should be a list.'
    assert image_width <= 1000, 'Image size is too large.'
    assert image_height <= 1000, 'Image size is too large.'

def test_load_training_data(self):
    np_ndarray_file = 'training.npy'
    load_array_d = CNN.load_array_data(np_ndarray_file)
    assert type(np_ndarray_file) == str, \
        'Wrong type. The np_ndarray_file should be a string.'

def test_data_information(self):
    input_data = []
    IMG = np.random.rand(134, 200)
    for i in range(1):
        input_data.append(['path', np.array(IMG), np.eye(4)[1]])
    np.save('training.npy', input_data)
    array_data = np.load('training.npy', allow_pickle=True)
    data_information = CNN.data_information(array_data)
    assert type(array_data) == np.ndarray, \
        'Wrong type. The array_data should be a numpy array.'

def test_plotting_data(self):
    input_data = []
    IMG = np.random.rand(134, 200)
    for j in range(5):
        input_data.append(['path', np.array(IMG), np.eye(4)[1]])
    i = 1
    ploting_d = CNN.ploting_data(input_data, i)
    assert i <= len(input_data), \
        'Invalid i. i should fall in the range of dataset size.'

def test_image_to_tensor(self):
    input_data = []
    IMG = np.random.rand(134, 200)
    for i in range(1):
        input_data.append(['path', np.array(IMG), np.eye(4)[1]])
    np.save('training.npy', input_data)
    array_data = np.load('training.npy', allow_pickle=True)
    image_width = 200
    image_height = 134
    image_to_tensor = CNN.image_to_tensor(array_data, image_width,
                                          image_height)
    assert type(array_data) == np.ndarray, \
        'Wrong type. The array_data should be a numpy array.'
    assert max(image_to_tensor) >= 1, \
	'normalization failed'

def test_type_to_tensor(self):
    input_data = []
    IMG = np.random.rand(134, 200)
    for i in range(1):
        input_data.append(['path', np.array(IMG), np.eye(4)[1]])
    np.save('training.npy', input_data)
    array_data = np.load('training.npy', allow_pickle=True)
    type_to_tensor = CNN.type_to_tensor(array_data)
    assert type(array_data) == np.ndarray, \
        'Wrong type. The array_data should be a numpy array.'

def test_data_separation(self):
    ratio_of_test = 0.2
    TRAIN = True
    TEST = False
    image_width = 200
    image_height = 134
    tensor_data = torch.randn(image_height, image_width).view(-1, 1, image_height, image_width)
    d_separation = CNN.data_separation(tensor_data, ratio_of_test,
                                       TRAIN, TEST)
    assert type(tensor_data) == torch.Tensor, \
        'Wrong type. The array_data should be a tensor.'
    assert 0 <= ratio_of_test <= 1, \
        'Invalid ratio. ratio_of_test should be in between 0 and 1.'
    assert TRAIN != TEST, 'Return only one type of sample in one time.'

def test_learning(self):
    image_width = 200
    image_height = 134
    training_sample_image = torch.randn(image_height, image_width).view(-1, 1, image_height, image_width)
    training_sample_type = torch.randn(1, 4)
    input_size = 1
    firstHidden = 8
    kernel_size = 5
    output_size = 4
    learning_rate = 0.001
    BATCH_SIZE = 10
    EPOCHS = 3
    learning = CNN.learning(training_sample_image, training_sample_type,
                            input_size, image_width, image_height,
                            firstHidden, kernel_size, output_size,
                            learning_rate, BATCH_SIZE, EPOCHS)
    assert len(training_sample_image) == len(training_sample_type), \
        'The number of image should equals to the number of label.'
    assert kernel_size <= 7, 'Maximum kernel_size is set as 7.'
    assert kernel_size % 2 == 1, 'kernel_size should be an odd integer'
    assert learning <= 1, 'loss must be less than 1'

def test_accuracy(self):
    image_width = 200
    image_height = 134
    testing_sample_image = torch.randn(image_height, image_width).view(-1, 1, image_height, image_width)
    testing_sample_type = torch.randn(1, 4)
    input_size = 1
    firstHidden = 8
    kernel_size = 5
    output_size = 4
    accuracy = CNN.accuracy(testing_sample_image, testing_sample_type,
                            input_size, image_width, image_height,
                            firstHidden, kernel_size, output_size)
    assert len(testing_sample_image) == len(testing_sample_type), \
        'The number of image should equals to the number of label'
    assert kernel_size <= 7, 'Maximum kernel_size is set as 7.'
    assert kernel_size % 2 == 1, 'kernel_size should be an odd integer'
    assert accuracy <= 1, 'Maximum accuracy is set as 1'

def test_type_prediction(self):
    k = 1
    input_size = 1
    image_width = 200
    image_height = 134
    path_List_training = ['testImage.png', 'training.npy']
    tensor_data = torch.randn(image_height, image_width).view(-1, 1, image_height, image_width)
    input_data = []
    IMG = np.random.rand(134, 200)
    for i in range(1):
        input_data.append(['path', np.array(IMG), np.eye(4)[1]])
    np.save('training.npy', input_data)
    array_data = np.load('training.npy', allow_pickle=True)
    firstHidden = 8
    kernel_size = 5
    output_size = 2
    detailed_information = True
    type_prediction = CNN.type_prediction(k, path_List_training,
                                          tensor_data, array_data,
                                          input_size, image_width,
                                          image_height, firstHidden,
                                          kernel_size, output_size,
                                          detailed_information)
    assert k == len(path_List_training) - 1, \
        'Incorrect number of folders/paths'
    assert len(path_List_training) == 2, \
        'Incorrect number of folders/paths.Classification/ Data Input must be 2; Bad, Passing'
    assert k <= 10, 'Too many folders/paths.'
    assert type(path_List_training) == list, \
        'path_List_predict should be a list'
    assert type(tensor_data) == torch.Tensor, \
        'Wrong type. The array_data should be a tensor.'
    assert type(array_data) == np.ndarray, \
        'Wrong type. The array_data should be a numpy.'
    assert image_width <= 1000, 'Image size is too large.'
    assert image_height <= 1000, 'Image size is too large.'
    assert kernel_size <= 7, 'Maximum kernel_size is set as 7.'
    assert kernel_size % 2 == 1, 'kernel_size should be an odd integer'
    assert type('processed.npy') == np.ndarray, \
        'Wrong type. The array_data should be a numpy array.'
