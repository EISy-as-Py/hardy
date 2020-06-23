## Instructions for using Convolutional Neural Network Config files


### Configuration file for CNN


For the recognition section of the package, there are two separate configuration files:

* __cnn_config.yaml__


_A configuration file which contains the hyperparameters to use in the single convolutional neural network.
The configuration file is easy to fill out and interact with._ 


__Note__: Make sure that the hyperparameters found in the config. file are also used in the cnn model


<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart_cnn_config.PNG" width=500 p align="center" />


### Configuration file for Hyperparameter Optimization

* __tuner_config.yaml__
    
A configuration file containing the hyperparamter search space for the tuning step. This should substitute the single cnn model. 
    
The first part deals with defining the tuner run :    
    

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart__tuner_config_run.PNG" width=500 p align="center" />


The second section, deals with the actual hyperparameter search space to use in the tuning operation :

<img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/Quickstart__tuner_config_space.PNG" width=500 p align="center" />