import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def make_training_data(IMG_SIZE, NOISE, IDEAL):
    """ Function that returns the imported data as numpy array

    Function that imports the images from the folders, add labels
    and then returns the imports as numpy array

    Parameters
    ----------
    IMG_SIZE : int
              an integer value indicating the image dimension for the
              import.
    NOISE : str
            a string indicating the folder in which noisy data is. Both
            absolute or relative path can be assigned.
    IDEAL : str
            a string indicating the folder in which ideal data is. Both
            absolute or relative path can be assigned.

    Returns
    -------
    training_data : numpy.array
                    numpy array containing the image data as vectors
                    and their corresponding label. It uses one-hot vector
                    to represent the label.
    """
    training_data = []
    LABELS = {NOISE: 0, IDEAL: 1}
    noisecount = 0
    idealcount = 0

    for label in LABELS:
        print(label)
        for f in tqdm(os.listdir(label)):
            if "png" or "jpg" in f:
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(img),
                                          np.eye(2)[LABELS[label]]])

                    if label == NOISE:
                        noisecount += 1
                    elif label == IDEAL:
                        idealcount += 1

                except Exception as e:
                    pass

    np.random.shuffle(training_data)
    np.save("training_data.npy", training_data)
    print('Noise:', noisecount)
    print('Ideal:', idealcount)

    training_data = np.load("training_data.npy", allow_pickle=True)
    print(len(training_data))

    return training_data


class Net(nn.Module):

    def __init__(self, IMG_SIZE, Input_Size, Hidden_layer, kernel):
        super().__init__()  # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(Input_Size, Hidden_layer, kernel)
        # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(Hidden_layer, Hidden_layer*2, kernel)
        # input is 32, bc the first layer output 32. Then we say the output
        # will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(Hidden_layer*2, Hidden_layer*3, kernel)

        x = torch.randn(IMG_SIZE, IMG_SIZE).view(-1, 1, IMG_SIZE, IMG_SIZE)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 32)  # flattening.
        self.fc2 = nn.Linear(32, 2)  # 512 in, 2 out bc we're doing 2 classes

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape, this flattens X
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return torch.sigmoid(x)


def optimize(net, loss_rate):
    optimizer = optim.Adam(net.parameters(), lr=loss_rate)
    loss_function = nn.MSELoss()

    return optimizer, loss_function


def data_split(training_data, frac, IMG_SIZE):
    X = torch.Tensor([i[0] for i in training_data]).view(-1, IMG_SIZE,
                                                         IMG_SIZE)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data])

    VAL_PCT = frac
    val_size = int(len(X)*VAL_PCT)
    print(val_size)

    train_X = X[:-val_size]
    train_y = y[:-val_size]

    test_X = X[-val_size:]
    test_y = y[-val_size:]
    print(len(train_X), len(test_X))

    return train_X, test_X, train_y, test_y


def train(train_X, train_y, net, IMG_SIZE, BATCH_SIZE, EPOCHS, optimizer,
          loss_function):

    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):  # 0 to the len
            # of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            # print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")

    return net


def test_accuracy(test_X, test_y, trained_network, IMG_SIZE):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = trained_network(test_X[i].view(-1, 1, IMG_SIZE,
                                                     IMG_SIZE))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1
    print("Accuracy: ", round(correct/total, 3))


def save_load_model(filename, network=None, save=None, load=None):
    if save:
        pickle.dump(network, open(filename+'.sav', 'wb'))
        return 0
    elif load:
        loaded_model = pickle.load(open(filename+'.sav', 'rb'))
        return loaded_model
