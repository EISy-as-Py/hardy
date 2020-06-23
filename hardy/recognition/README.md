## Instructions for using Convolutional Neural Network Config files

Each configuration file in the recognition folder provides the example Hyperparameter space over which Neural Network model is built

### Configuration file for CNN

This file provides the input information along with the Hyperparameter space for Neural Network.

* __cnn_config.yaml__


_A configuration file which contains the hyperparameters to use in the single convolutional neural network.
The configuration file is easy to fill out and interact with._


__Note__: Make sure that the hyperparameters found in the config. file are also used in the cnn model


<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart_cnn_config.PNG" width=500 p align="center" />

Currently supported keys for the <code>cnn_config.yaml</code> includes:

<code>
* kernel_size

* epochs

* activation

* input_shape

* filter_size

* num_classes

* learning_rate

* patience
</code>

__Note__:The detailed information about the options can be found in the config file itself.

### Configuration file for Hyperparameter Optimization

* __tuner_config.yaml__
    
A configuration file containing the hyperparamter search space for the tuning step. This should substitute the single cnn model. 
    
The first part deals with defining the tuner run :    
    

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart__tuner_config_run.PNG" width=500 p align="center" />


The second section, deals with the actual hyperparameter search space to use in the tuning operation :

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart__tuner_config_space.PNG" width=500 p align="center" />