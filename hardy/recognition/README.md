## Instructions for using Convolutional Neural Network Configuration files

Each configuration file in the recognition folder provides the example Hyperparameter space over which Neural Network model is built. The configuration file can be placed anywhere in the system and relative path must be passed as argument in <code>run_hardy.hardy_multi_transform</code> module

### Configuration file for CNN

This file provides the input information along with the Hyperparameter space for Neural Network.

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart_cnn_config.PNG" width=500 p align="right" />

* __cnn_config.yaml__

_A configuration file which contains the hyperparameters to use in the single convolutional neural network.
The configuration file is easy to fill out and interact with._


__Note__: Make sure that the hyperparameters found in the config. file are also used in the cnn model


Currently supported keys for the <code>cnn_config.yaml</code> includes:

```
-> kernel_size

-> epochs

-> activation

-> input_shape

-> filter_size

-> num_classes

-> learning_rate

-> patience
```

__Note__: All this information must be entered to successfully execute the Machine Learning Step. The detailed information about the options for keys can be found in the config 
<a href=https://github.com/EISy-as-Py/hardy/blob/master/hardy/recognition/cnn_config.yaml>file</a> itself.

<hr>

### Configuration file for Hyperparameter Optimization

* __tuner_config.yaml__
    
A configuration file containing the hyperparamter search space for the tuning step. This should substitute the single cnn model. 

The first part deals with defining the tuner run:   

For definition of tuner run, following keys are currently supported:

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart__tuner_config_run.PNG" width=450 p align="right" />


```
-> num_classes

-> epochs

-> patience

-> input_shape

-> max_trials

-> exec_per_trial

-> search_function
```

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart__tuner_config_space.PNG" width=400 p align="right" />

The second section, deals with the actual hyperparameter search space to use in the tuning operation:

For hyperparameters search space, following keys are currently supported:
```
-> layers

-> filters

-> kernel_size

-> activation

-> pooling

-> optimizer

-> learning_rate
```

__Note__: All this information must be entered to successfully execute the Hyperparameter Step. The detailed information about the options for keys can be found in the config <a href=https://github.com/EISy-as-Py/hardy/blob/master/hardy/recognition/tuner_config.yaml>file</a> itself.
