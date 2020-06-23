## Instructions for using Convolutional Neural Network Config files


## Config for CNN


For the classifier section of the package, there are two separate configuration files:

* __cnn_config.yaml__


_A configuration file which contains the hyperparameters to use in the single convolutional neural network.
The configuration file is easy to fill out and interact with._ 


__Note__: Make sure that the hyperparameters found in the config. file are also used in the cnn model


<img src="../../doc/Images/quickstart_cnn_config.PNG" width=500 p align="center" />


* __tuner_config.yaml__
    
A configuration file containing the hyperparamter search space for the tuning step. This should substitute the single cnn model. 
    
The first part deals wiht defining the tuner run :    
    
<img src="./doc/Images/Quickstart__tuner_config_run.PNG" width=500 p align="center" />


The second section, deals with the actual hyperparameter search space to use in the tuning operation :

<img src="../../doc/Images/Quickstart__tuner_config_space.PNG" width=500 p align="center" />