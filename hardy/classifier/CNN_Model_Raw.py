# Following are all included in environment.yml

import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#%matplotlib inline

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
        path_list_training: A list containing the path of training folder.
                            One index for one path only.
                            Last index is the nparray file name (XXX.npy).
        image_width: The target width after resize
        image_height: The target height after resize
        """
        
        path_list = path_List_training
        countImage_Training = [0, 0, 0, 0, 0, 0, 0]
        training_data = []
        # Iterate the directory
        print(len(path_list))
        for label in range(len(path_list)):
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
                    # Label the image with np.eye() matrix.
                    training_data.append([path, np.array(img),
                                          np.eye(k-1)[label]])
                    for i in range(k):
                        if label == i:
                            countImage_Training[i] += 1

        np.random.shuffle(training_data)
        np.save(path_list[-1], training_data)
        for i in range(len(path_list)-1):
            print(path_List_training[i], ":", countImage_Training[i])
        return training_data

    def DataImporter_Predict(self, k, path_List_predict,
                             image_width, image_height):
        
        """
        Import the testing image file (.png) into the model.
        Parameters
        ----------
        k: The total number of path(folder)
           (Setting the maximum value equal 10 by defult)
        path_list_training: A list containing the path of training folder.
                            One index for one path only.
                            Last index is the nparray file name (XXX.npy).
        image_width: The target width after resize
        image_height: The target height after resize
        """
        
        path_list = path_List_predict
        countImage_Predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        training_data = []
        # Iterate the directory
        for label in range(len(path_list)):
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
        for i in range(len(path_list)):
            print(path_List_predict[i], ":", countImage_Predict[i])

def Build_Data(Training, k, path_list, image_width, image_height):
    
    '''This will allow us to use EISData() to preprocess train/test data.
    Parameters
    ----------
    Training = True (Class.DataImporter_Training) will be used/ 
    False (vice versa)
    k = the total number of path
    path_list = [string of folder path on local drive]
    image_width/height for rescaling  
    Returns
    --------
    training data [path,numpy array for images, label(if Training=True)]
    '''
    
    Class = EISData()
    if Training is True:
        Class.DataImporter_Training(k, path_list, image_width, image_height)
    else:
        Class.DataImporter_Predict(k, path_list, image_width, image_height)

def load_training_data(np_ndarray_file):
    
    """
    Load the data from the .npy file to check if all the images
    have been in the program.

    Parameter
    ----------
    np_ndarray_file: The XXX.npy file name.
                     Should be identical to the last index in path list

    Returns
    ----------
    training_data:  the dataset expressed in numpy array form.
                    type -> numpy.ndarray

    """
    
    training_data = np.load(np_ndarray_file, allow_pickle=True)
    return training_data

def data_information(training_data):
    
    """
    Check the size of image and dataset.

    Parameters
    ----------
    training_data: the data loading from "eis_training_data.npy"

    """
    
    print("Type of training_data:", type(training_data))
    print("Size of training_data:", len(training_data))
    print("Size of image(after rescale):", training_data[0][1].shape[1],
          "x", training_data[0][1].shape[0])

def plotting_data(training_data, k):
    
    """
    Show the assigned image with matplotlib package.

    Parameters
    ----------
    training_data: the data loading from "eis_training_data.npy"
    k:  assign one image in training_data to show.
        k should fall in the range of dataset size.
        Rang: 0-size of training data

    """
    
    print(training_data[k][0])
    plt.imshow(training_data[k][1])
    plt.show

class Net(nn.Module):
    
    """Convolutional Neural Network Model"""
    
    def __init__(self, input_size, image_width, image_height,
                 firstHidden, kernel_size, output_size):
        
        """

        Parameters
        ----------
        input_size: 1
        image_width: The width of input images.
                     this is provided from the data_information function
        image_height: The width of input images.
                     this is provided from the data_information function
        firstHidden: The size of first hidden layer.
                     The size of next layer will be twice of the current layer
                     Ex: 1st is 8, 2nd will be 16, 3rd will be 24 and so on.
                     Number of hidden layer is set as 4 by default.
        kernel_size: It will form a subwindom with size of kernel to scan over
                     the original image.
                     kernel_size must be an odd integer,
                     usually not larger than 7
        output_size: The number of final target category.

        """
        
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_size, firstHidden, kernel_size)
        self.conv2 = nn.Conv2d(firstHidden, firstHidden*2, kernel_size)
        self.conv3 = nn.Conv2d(firstHidden*2, firstHidden*4, kernel_size)
        self.conv4 = nn.Conv2d(firstHidden*4, firstHidden*8, kernel_size)
        # Get size
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
        x = self.convs(x)
        conv_to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        # Flatten the data
        xF = x.view(-1, conv_to_linear)
        # put into the first fully connected layer
        output = torch.sigmoid(self.fc1(xF))
        output = self.fc2(output)
        return F.softmax(output, dim=1)

def data_separation(data, ratio_of_testing, TRAIN):
    
    """
    Separate the training and testing data.
    Parameters
    -------------
    data: A tensor. Tranformed image numpy array.
    ratio_of_testing. Float. 0.2 = 20% of data is stored as testing data
    Train: True/False
    """
    
    VAL_PCT = ratio_of_testing
    val_size = int(len(data)*VAL_PCT)

    if TRAIN is True:
        train_data = data[:-val_size]
        print("Training Samples:", len(train_data))
        return train_data
    test_data = data[-val_size:]
    print("Testing Samples:", len(test_data))
    return test_data

def image_to_tensor(training_data, image_height, image_width):
    
    """Transform the array image into tensor."""
    
    X = torch.Tensor([i[1] for i in training_data]
                     ).view(-1, image_height, image_width)
    return X/255.  #  normalize X

def type_to_tensor(training_data):
    
    """Transform the array type into tensor."""
    
    y = torch.Tensor([i[2] for i in training_data])
    return y

def learning(train_data1, train_data2, input_size, image_width, image_height,
             firstHidden, kernel_size, output_size, learning_rate, BATCH_SIZE,
             EPOCHS):
    
    """
    parameter
    ---------
    train_data1 = X_train [image data]
    train_data2 = y_train [type data]
    input_size = 1
    image_width, image_height = same as the ones defined for preprocessing
    firstHidden = 8, typically
    kernel_size = 4
    output_size = 2 [0=bad, 1=pass]
    learning_rate = 0.001 at default for pytorch
    BATCH_SIZE = 10 the number of images used for training at a time
    EPOCHS = 1
    """
    
    optimizer = optim.Adam(Net(input_size, image_width, image_height,
                               firstHidden, kernel_size, output_size
                               ).parameters(), lr=learning_rate)
    loss_function = nn.L1Loss()  #  L1Loss() simple loss function

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_data1), BATCH_SIZE)):
            batch_data1 = train_data1[i:i+BATCH_SIZE].view(-1, 1,
                                                           image_height,
                                                           image_width)
            batch_data2 = train_data2[i:i+BATCH_SIZE]

            Net(input_size, image_width, image_height, firstHidden,
                kernel_size, output_size).zero_grad()
            outputs = Net(input_size, image_width, image_height, firstHidden,
                          kernel_size, output_size)(batch_data1)
            loss = loss_function(outputs, batch_data2)
            loss.backward()
            optimizer.step()

        print(loss)

def accuracy(test_data1, test_data2, input_size, image_width, image_height,
             firstHidden, kernel_size, output_size):
    
    """
    This function tells how well the predictions are made based 
    on the correctness.
    Parameters:
    --------------
    test_data1 = tensor. From image_to_tensor
    test_data2 = tensor. From type_to_tensor
    input_size,image_width,image_height,firstHidden,kernel_size,output_size
    same as the function call for learning
    """
    
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_data1))):
            real_type = torch.argmax(test_data2[i])
            net_out_train = Net(input_size, image_width, image_height,
                                firstHidden, kernel_size, output_size
                                )(test_data1[i].view(-1, 1, image_height,
                                                     image_width))[0]
            predicted_type = torch.argmax(net_out_train)

            if predicted_type == real_type:
                correct += 1
            total += 1

    print("Accuracy:", round(correct/total, 3))

def type_prediction(k, path_List_training, tensor_data, array_data,
                    input_size, image_width, image_height, firstHidden,
                    kernel_size, output_size, detailed_information):
    
    """
    Predict which type the input image is and print out the total number of
    each type.
    (Optional) Print out the predicted type and file name for each image.
    Parameters
    ----------
    k
    path_List_training : Same as the parameters of nn model
    tensor_data: from the return of image_to_tensor() function.
    array_data: from the return of load_array_data() function.
    input_size : 1
    image_width: The target width after resize
    image_height: The target height after resize
    firstHidden: The size of first hidden layer.
    kernel_size: It will form a subwindom with size of kernel to scan over
                 the original image.
    output_size: The number of final target type.
    detailed information: Show the predicted type and file
                          name for each image or not
    """
    
    passing_data = []
    countImage_predicted_type = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(Input_data)):
        net_out_predict = Net(input_size, image_width, image_height,
                              firstHidden, kernel_size, output_size)(Input_data[i].view(-1, 
                              1, image_height, image_width))[0]
        predicted_type = torch.argmax(net_out_predict)
        for Type in range(k):
            if predicted_type == 0:
                countImage_predicted_type[Type] += 1
                # Print out the detailed information.
                if detailed_information is True:
                    #  The following messages are optional.
                    print("Warning!Type Prediction:", path_List_training[Type])
                    print("Path and File Name", array_data[i][0])
            else:
                countImage_predicted_type[Type] += 1
                passing_data.append([array_data[i][0], array_data[i][1]])
                
    for i in range(len(path_List_training)-1):
        print(path_List_training[i], ":", countImage_predicted_type[i])
    
    np.save('processed.npy', passing_data)
    
