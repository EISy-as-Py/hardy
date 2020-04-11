"""Import necessary package"""
import os
import cv2
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EISDataImport():
    """Data Import and Pre-Processing"""

    def DataImporter_Training(self, k, path_List_training,
                              image_width, image_height):
        """
        Import the training image file (.png) into the model.

        Parameters
        ----------
        k: The total number of the type.
           (Setting the maximum value equal 7 by defult)
        path_list_training: A list containing the path of training images.
                            One index for one path only.
                            Last index is the nparray file name (XXX.npy).
        image_width: The target width after resize
        image_height: The target height after resize

        """
        path_list = path_List_training
        countImage_Training = [0, 0, 0, 0, 0, 0, 0]
        training_data = []
        # Iterate the directory
        for label in range(len(path_list)-1):
            print(path_list[label])
            # Iterate all the image within the directory, f -> the file name
            for f in tqdm(os.listdir(path_list[label])):
                # Get the full path to the images
                path = os.path.join(path_list[label], f)
                if "png" in path:
                    # Read images in the given path and turn into nparray.
                    # Convert the iimage to gray scale (optional)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (image_width, image_height))
                    # Label the image with np.eye() matrix.
                    training_data.append([path, np.array(img),
                                          np.eye(k)[label]])
                    for i in range(k):
                        if label == i:
                            countImage_Training[i] += 1

        np.random.shuffle(training_data)
        np.save(path_list[-1], training_data)
        for i in range(len(path_list)-1):
            print(path_List_training[i], ":", countImage_Training[i])

    def DataImporter_Predict(self, k, path_List_predict,
                             image_width, image_height):
        """
        Import the testing image file (.png) into the model.

        Parameters
        ----------
        k: The total number of path(folder)
           (Setting the maximum value equal 10 by defult)
        path_list_predict: A list containing the path of random image to 
                           be predicted.
                           One index for one path only.
                           Last index is the nparray file name (XXX.npy).
        image_width: The target width after resize
        image_height: The target height after resize

        """
        path_list = path_List_predict
        countImage_Predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        training_data = []
        # Iterate the directory
        for label in range(len(path_list)-1):
            print(path_list[label])
            # Iterate all the image within the directory, f -> the file name
            for f in tqdm(os.listdir(path_list[label])):
                # Get the full path to the image
                path = os.path.join(path_list[label], f)
                if "png" in path:
                    # Read images in the given path and turn into nparray.
                    # Convert the iimage to gray scale (optional)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (image_width, image_height))
                    training_data.append([path, np.array(img)])
                    # Count the number of image
                    for i in range(k):
                        if label == i:
                            countImage_Predict[i] += 1

        np.random.shuffle(training_data)
        np.save(path_list[-1], training_data)
        for i in range(len(path_list)-1):
            print(path_List_predict[i], ":", countImage_Predict[i])


def Build_Data(Training, Predict, k, path_list, image_width, image_height):
    """
    Determine the type of data to build.

    Parameters
    ----------
    Training: True for building the training data
    Predict: True for building the predict data
    k: Depends on training/predict switch:
       The total number of type/path(folder).
       (Setting the maximum value equal 7/10 by defult)
    path_list_training: A list containing the path of training folder.
                        One index for one path only.
                        Last index is the nparray file name (XXX.npy).
    image_width: The target width after resize
    image_height: The target height after resize

    """
    Class = EISDataImport()
    if Training is True:
        Class.DataImporter_Training(k, path_list, image_width, image_height)
    if Predict is True:
        Class.DataImporter_Predict(k, path_list, image_width, image_height)


def load_array_data(np_ndarray_file):  # Data Status Check
    """
    Load the data from the .npy file to check if all the images
    have been in the program.

    Parameter
    ----------
    np_ndarray_file: The XXX.npy file name.
                     Should be identical to the last index in path list

    Return
    ----------
    training_data:  The dataset expressed in numpy array form.
                    type -> numpy.ndarray

    """
    array_data = np.load(np_ndarray_file, allow_pickle=True)
    return array_data


def data_information(array_data):
    """
    Check the size of image and dataset.

    Parameters
    ----------
    array_data: the data in nparray form loading from "XXX.npy" file.

    """
    print("Type of input_data:", type(array_data))
    print("Size of imput_data:", len(array_data))
    print("Size of image(after rescale):", array_data[0][1].shape[1],
          "x", array_data[0][1].shape[0])


def plotting_data(input_data, i):
    """
    Show the assigned image with matplotlib package.

    Parameters
    ----------
    input_data: the data in nparray form loading from "XXX.npy" file.
    i:  An arbitrary number to assign one image in input_data to show.
        Should fall in the range of dataset size.

    """
    print(input_data[i][0])  # Print out the file name (path).
    plt.imshow(input_data[i][1])
    plt.show()


class Net(nn.Module):
    """Convolutional Neural Network Model"""
    def __init__(self, input_size, image_width, image_height,
                 firstHidden, kernel_size, output_size):
        """
        Parameters
        ----------
        input_size: Setting as 1 for gray scale image.
        image_width: The target width after resize
        image_height: The target height after resize
        firstHidden: The size of first hidden layer.
                     The size of next layer will be twice of the current layer
                     Ex: 1st is 8, 2nd will be 16, 3rd will be 24 and so on.
                     Number of hidden layer is set as 4 by default.
        kernel_size: It will form a subwindom with size of kernel to scan over
                     the original image.
                     Kernel_size must be an odd integer,
                     Usually not larger than 7.
        output_size: The number of final target type.
                     (For the training part, it should be identical to
                     the k value in DataImporter_Training function())

        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_size, firstHidden, kernel_size)
        self.conv2 = nn.Conv2d(firstHidden, firstHidden*2, kernel_size)
        self.conv3 = nn.Conv2d(firstHidden*2, firstHidden*4, kernel_size)
        self.conv4 = nn.Conv2d(firstHidden*4, firstHidden*8, kernel_size)
        #
        x = torch.randn(image_height, image_width).view(-1, 1, image_height,
                                                        image_width)
        conv_to_linear = self.last_conv_neuron(x)

        self.fc1 = nn.Linear(conv_to_linear, 64)
        self.fc2 = nn.Linear(64, output_size)

    def last_conv_neuron(self, x):
        """
        Calculate how many neurons that the last convolutional layer will
        connect to the linear hidden layer

        Parameters
        ----------
        x: a random torch tensor with size (-1, 1, image_height, image_width)
        Ex: x = torch.randn(image_height, image_width
                            ).view(-1, 1, image_height, image_width)
        """
        x = self.convs(x)
        conv_to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return conv_to_linear

    def convs(self, x):
        """
        Put the image into the convolutional hidden layer. Scan over the
        original image to and use the max pooling function (with size 2) to
        determine the one number to represent the sub-image.

        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        return x

    def forward(self, x):
        """
        Determine the order that image pass through the neural network model.

        """
        x = self.convs(x)
        conv_to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        # Flatten the data
        xF = x.view(-1, conv_to_linear)
        # Put into the first fully connected layer
        output = F.relu(self.fc1(xF))
        output = self.fc2(output)
        return F.softmax(output, dim=1)


def image_to_tensor(array_data, image_width, image_height):
    """
    Transform the array image into tensor.

    Parameters
    ----------
    array_data: the data in nparray form loading from "XXX.npy" file.
    image_width: The target width after resize
    image_height: The target height after resize

    Return
    ----------
    tensor_image: Image data in tensor-form.

    """
    tensor_image = torch.Tensor([i[1] for i in array_data]
                                ).view(-1, image_height, image_width)
    return tensor_image


def type_to_tensor(array_data):
    """
    The type here means the labels which were put with the images
    while importing
    Transform the array type into tensor.
    The function should be used only in the training part.

    Parameters
    ----------
    array_data: the data in nparray form loading from "XXX.npy" file.

    Return
    ----------
    tensor_type: Types data in tensor-form.

    """
    tensor_type = torch.Tensor([i[2] for i in array_data])
    return tensor_type


def data_separation(tensor_data, ratio_of_testing, TRAIN, TEST):
    """
    Separate the training and testing data.

    Parameters
    ----------
    tensor_data: the data in tensor form
    ratio_of_testing: ratio for the testing data
    TRAIN: determine which size of data will be printed out
           if TRIAN is True, print out the training sample size;
           otherwise, print out the testing sample size
    """
    VAL_PCT = ratio_of_testing
    val_size = int(len(tensor_data)*VAL_PCT)

    if TRAIN is True:
        training_sample = tensor_data[:-val_size]
        print("Training Samples:", len(training_sample))
        return training_sample
    if TEST is True:
        testing_sample = tensor_data[-val_size:]
        print("Testing Samples:", len(testing_sample))
        return testing_sample


def learning(training_sample_image, training_sample_type, input_size,
             image_width, image_height, firstHidden, kernel_size, output_size,
             learning_rate, BATCH_SIZE, EPOCHS):
    """
    Put the training sample into to the neural network model to learn.

    Parameters
    ----------
    training_sample_image: The tensor-form images used to train the model

    training_sample_type: The tensor-form types used to train the model

    ---Same as the parameters of nn model---
    input_size: Setting as 1 for gray scale image.
    image_width: The target width after resize
    image_height: The target height after resize
    firstHidden: The size of first hidden layer.
    kernel_size: It will form a subwindom with size of kernel to scan over
                  the original image.
    output_size: The number of final target type.

    learning_rate: The learning rate controls how quickly the model is adapted
                   to the problem (often in the range between 0.0 and 1.0.)
                   
    BATCH_SIZE: Number of training examples utilized in one iteration.
    EPOCHS: Number of iterations in the whole training process
    """
    optimizer = optim.Adam(Net(input_size, image_width, image_height,
                               firstHidden, kernel_size, output_size
                               ).parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(training_sample_image), BATCH_SIZE)):
            batch_image = training_sample_image[i:i+BATCH_SIZE
                                                ].view(-1, 1, image_height,
                                                       image_width)
            batch_type = training_sample_type[i:i+BATCH_SIZE]

            Net(input_size, image_width, image_height, firstHidden,
                kernel_size, output_size).zero_grad()
            outputs = Net(input_size, image_width, image_height, firstHidden,
                          kernel_size, output_size)(batch_image)
            loss = loss_function(outputs, batch_type)
            loss.backward()
            optimizer.step()

        print(loss)


def accuracy(testing_sample_image, testing_sample_type, input_size,
             image_width, image_height, firstHidden, kernel_size,
             output_size):
    """
    Test the predicting accuracy for the learning function

    Parameters
    ----------
    testing_sample_image: The tensor-form images used to test model accuracy
    testing_sample_type: The tensor-form types used to test model accuracy

    ---Same as the parameters of nn model---
    input_size: Setting as 1 for gray scale image.
    image_width: The target width after resize
    image_height: The target height after resize
    firstHidden: The size of first hidden layer.
    kernel_size: It will form a subwindom with size of kernel to scan over
                  the original image.
    output_size: The number of final target type.

    """
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(testing_sample_image))):
            real_type = torch.argmax(testing_sample_type[i])
            net_out_train = Net(input_size, image_width, image_height,
                                firstHidden, kernel_size, output_size
                                )(testing_sample_image[i].view(-1, 1,
                                                               image_height,
                                                               image_width
                                                               ))[0]
            predicted_type = torch.argmax(net_out_train)

            if predicted_type == real_type:
                correct += 1
            total += 1

    predicting_accuracy = round(correct/total, 3)
    print("Accuracy:", predicting_accuracy)
    return predicting_accuracy


def type_prediction(k, path_List_training, tensor_data, array_data,
                    input_size, image_width, image_height, firstHidden,
                    kernel_size, output_size, detailed_information, j):
    """
    Predict which type the input image is and print out the total number of
    each type.
    (Optional) Print out the predicted type and file name for each image.

    Parameters
    ----------
    k: The total number of "TRAINING" folder.
       Must be identical to the first parameter in "DataImporter_Training" 
       function.
    path_list_training: Must be identical to the second parameter 
                        in "DataImporter_Training function".
                        A list containing the path of "TRAINING" folder.
                        One index for one path only.
                        Last index is the nparray file name (XXX.npy).
    tensor_data: from the return of image_to_tensor() function.
    array_data: from the return of load_array_data() function.
    ---Same as the parameters of nn model---
    input_size: Setting as 1 for gray scale image.
    image_width: The target width after resize
    image_height: The target height after resize
    firstHidden: The size of first hidden layer.
    kernel_size: It will form a subwindom with size of kernel to scan over
                 the original image.
    output_size: The number of final target type.

    detailed information: Show the predicted type and file
                          name for each image or not.

    j: Determine how many images will print out the detailed information.

    """
    countImage_predicted_type = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(tensor_data)):
        net_out_predict = Net(input_size, image_width, image_height,
                              firstHidden, kernel_size, output_size
                              )(tensor_data[i].view(-1, 1, image_height,
                                                    image_width))[0]
        predicted_type = torch.argmax(net_out_predict)
        for Type in range(k):
            if predicted_type == Type:
                countImage_predicted_type[Type] += 1
                # Print out the detailed information.
                if i < j:                    
                    if detailed_information is True:
                        print("Type Prediction:", path_List_training[Type])
                        print("Path and File Name", array_data[i][0])

    for i in range(len(path_List_training)-1):
        print(path_List_training[i], ":", countImage_predicted_type[i])
