[![Build Status](https://travis-ci.org/EISy-as-Py/hardy.svg?branch=master&kill_cache=1)](https://travis-ci.org/EISy-as-Py/hardy)
[![Coverage Status](https://coveralls.io/repos/github/EISy-as-Py/hardy/badge.svg?branch=master&kill_cache=1)](https://coveralls.io/github//EISy-as-Py/hardy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/hardy/badge/?version=latest)](https://hardy.readthedocs.io/en/latest/?badge=latest)
<img src=https://github.com/EISy-as-Py/hardy/blob/master/doc/images/EIS_Formats.PNG width=400 p align="right">

# Project HARDy
 
 _"Handling Arbitrary Recognition of Data, y not?"_
A package to assist in discovery, research, and classification of YOUR data, no matter who you are! 

## Installation:
Use the following steps to install the HARDy package:
* In the terminal run, <code>git clone https://github.com/EISy-as-Py/hardy.git<code>
* 'pip' and 'conda' install is coming soon 

WORK IN PROGRESS - Please feel free to provide feedback and expand on our vision!
-----------------------------------------------------------------
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
 
 ------------------------------------------------------------------
 ## Modules and Goals:
 * __handling.py__         :  Functions related to configuration, importing/exporting, and other sorts of back-end useful tasks.
 * __arbitrage.py__        :  Data Pre-Analysis, Transformations, and other preparation to be fed into the learning algorythm.
 * __recognition.py__      :  The real Meat-and-Potatoes! Setup, Training, and output from whatever ML Algorythm(s?) we use.
 * __data_reporting.py__   :  Output and reporting of any/all discoveries, maybe with an eye to long-term evolution and improving the quality and speed of the program.
 * __yNot.py__             :  Home for everything else that seems useful but doesn't fall into one of the other functions! (Probably will contain any/all of the User Interface tools we may or may not work on).
 
 
 
 
