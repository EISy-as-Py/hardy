[![Build Status](https://app.travis-ci.com/EISy-as-Py/hardy.svg?branch=master)](https://app.travis-ci.com/EISy-as-Py/hardy)
[![Coverage Status](https://coveralls.io/repos/github/EISy-as-Py/hardy/badge.svg?branch=master)](https://coveralls.io/github/EISy-as-Py/hardy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/hardy/badge/?version=latest)](https://hardy.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/pozzorg/hardy/badges/platforms.svg)](https://anaconda.org/pozzorg/hardy)
[![Anaconda-Server Badge](https://anaconda.org/pozzorg/hardy/badges/installer/conda.svg)](https://conda.anaconda.org/pozzorg)
[![Anaconda-Server Badge](https://anaconda.org/pozzorg/hardy/badges/license.svg)](https://anaconda.org/pozzorg/hardy)


[![DOI](https://joss.theoj.org/papers/10.21105/joss.03829/status.svg)](https://doi.org/10.21105/joss.03829)

<img src=https://github.com/EISy-as-Py/hardy/blob/master/doc/images/EIS_Formats.PNG width=400 p align="right">

# Project HARDy

 _"HARDy: Handling Arbitrary Recognition of Data in python"_
A package to assist in discovery, research, and classification of YOUR data, no matter who you are!

## Project Objective

Numerical and visual transformation of experimental data to improve its classification and cataloging

This project was part of DIRECT Capstone Project at University of Washington and was presented at the showcase, follow this
<a href=https://prezi.com/view/5ugf5HyDxZevQlOHmuyO/>link </a>  for the presentation

## Requirements:
Package HARDy has following main dependencies:
1. Python = 3.7
2. Tensorflow = 2.0

The detailed list of dependencies is reflected in the <a href=https://github.com/EISy-as-Py/hardy/blob/master/environment.yml><code>environment.yml</code></a> file

## Installation:
The package HARDy can be installed using following command:

<code>conda install -c pozzorg hardy </code>

Alternatively, you can also install it using the GitHub repository in following steps:

*Please note that currently v1.0 is the most stable release

1. In your terminal, run <code>git clone https://github.com/EISy-as-Py/hardy.git</code>
2. Change the directory to hardy root directory, by running <code>cd hardy</code> 
3. Run <code>git checkout v1.0</code>
4. Run <code>python setup.py install</code>
5. To check installation run, <code>python -c "import hardy"</code> in your terminal

For other methods of installation like using environment file and installation using pip, please visit <a href=https://hardy.readthedocs.io/en/latest/installation.html>Installation<a> page.

## Usage:

HARDy uses Keras for training Convolutional Neural Network & Keras-tuner for the hyperparameter optimization. The flow of information is shown in image below:

<p align="center"><img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/HARDy_diagram.png" width=700 alt="information flow of how the package works"/></p>

An example jupyter notebook to run HARDy using single script is available at this link <a href=https://github.com/EISy-as-Py/hardy/blob/master/doc/examples/example_HARDy_script.ipynb>Example Notebook</code></a>

To perform various transformations, training Neural Network and Hyperparameter Optimization, Hardy utilizes following <code>.yaml</code> configuration files:

* <a href=https://github.com/EISy-as-Py/hardy/blob/master/hardy/arbitrage/README.md>tform_config.yaml</a>
* <a href=https://github.com/EISy-as-Py/hardy/blob/master/hardy/recognition/README.md>cnn_config.yaml</a>
* <a href=https://github.com/EISy-as-Py/hardy/blob/master/hardy/recognition/README.md>tuner_config.yaml</a>

The instructions for modifying or writing your own configuration file can be accessed by clicking on the configuration files listed above.

The notebooks and documentations can also be accessed at this link <a href=https://hardy.readthedocs.io/en/latest/index.html>Documentations</a>

## Visualization
 In order to increase the density of data presented to the convolutional neural network and add a visual transformation of the data, we adopted a new plotting technique that takes advantage of how images are read by computers. Using color images, we were able to encode the experimental data in the pixel value, using different series per each image channel. The results are data- dense images, which are also pretty to look at.

 <p align="center"><img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/data_visualization.PNG" width=700 alt=" details on the proposed visual transformation to increased the images data density"/></p>


## Mission:
We have been commissioned by Professor Lilo Pozzo to create a new tool for research and discovery, For her lab and for high throughput researchers everywhere.
Our vision of the final product:
 * A package which can approach any large, labeled dataset (such as those familiar to High Throughput Screening (HTS) researchers).
 * Perform a (procedurally generated and data-guided) wide array of transformations on the data to produce completely novel ways of examining the data, maybe not Human-Readable but in a certainly machine-readable format.
 * Train "A Machine Learning Algorithm" (We currently focus on Visual-Processing CNNs but are open to anything!) to classify the existing labled data based on each of the aforementioned transformations.
 * Report back to the user:
    * Which versions of the Model/Algorithm worked best?
    * Which transformations appeared the most useful? (AKA were used across many of the most successful models)
    * What Data "Fingerprints" should we pay the most attention to?
 * Present a User Interface, to allow non-programmers to interact with and use the chosen classifier(s?) in their work.

 ## Use Cases:
 The package is designed to deal with a diverse set of labeled data. These are some of the use cases we see benefitting from using the _HARDy_ package.

 <p align="center"><img src="https://github.com/EISy-as-Py/hardy/blob/master/doc/images/use_cases.PNG" width=500 alt="possible use cases for the HARDy package"/></p>


 ## Modules Overview:
 * __handling.py__         :  Functions related to configuration, importing/exporting, and other sorts of back-end useful tasks.
 * __arbitrage.py__        :  Data Pre-Analysis, Transformations, and other preparation to be fed into the learning algorithm.
 * __recognition.py__      :  Setup, training and testing of single convolutional neural network (CNN) or hyperparameters optimization for CNNs.
 * __data_reporting.py__   :  Output and reporting of any/all results. Tabular summary of runs, visual performance comparison, as well as parallel coordinate plots and feature maps

 ## Community Guidlines:
 We welcome the members of open-source community to extend the functionalities of HARDy, submit feature requests and report bugs.
 
 ### Feature Request:
 If you would like to suggest a feature or start a discussion on possible extension of HARDy, please feel free to <a href="https://github.com/EISy-as-Py/hardy/issues/new/choose">raise an issue</a>
 
 ### Bug Report:
 If you would like to report a bug, please follow <a href="https://github.com/EISy-as-Py/hardy/issues/new/choose">this link</a>
 
 ### Contributions:
 If you would to contribute to HARDy, you can fork the repository, add your contribution and generate a pull request. The complete guide to make contributions can be found at this <a href="https://github.com/EISy-as-Py/hardy/blob/master/CONTRIBUTIONS.md">link</a>

 ## Acknowledgment

 Maria Politi acknowledges support from the National Science Foundation through NSF-CBET grant 1917340
