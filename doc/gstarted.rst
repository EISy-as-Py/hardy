Getting Started
===============
:code:`Hardy`, by default, is configured to take minimal inputs
from the user and perform numerical and visual transformations 
on its own. The numerical transformations follow rules defined
by the user in a :code:`.yaml` configuration files. The user can
perform either hyperparamter search to evaluate best hyperparameters
or run a simple convolutional neural network (CNN).
The hyperparameter space for both tuning session
or :code:`CNN` can be defined in a :code:`.yaml` configuration
file. The guide to write configuration files is available at
`Guide to write configuration files 
<https://hardy.readthedocs.io/en/latest/examples/How_to_write_Configuration_files.html>`_

Data Preparation
----------------
:code:`HARDy` is configured to input :code:`.csv` files only. Before
starting your :code:`HARDy` run, make sure the data files are only in
:code:`.csv` format. Moreover, the :code:`.csv` files must have a header
of same length. 

The wrapper function, :code:`run_hardy`, takes care of all the numerical
and visual transformations along with hyperparameter tuning and CNN runs.
The example script to run HARDy is as follows:

Importing HARDy library
-----------------------

``import hardy.run_hardy as run``

Defining path variables
-----------------------
Defining the path to :code:`.csv` files::

    raw_data_path = 'path/to/raw/data/'

Defining the path to numerical configuration file::

    tform_config_path = './hardy/arbitrage/tform_config.yaml'

Defining the path to tuner or CNN configuration::

    classifier_config_path = './hardy/recognition/'

Executing hardy_main
--------------------

``run.hardy_main(raw_data_path, tform_config_path, classifier_config_path, batch_size=64,
scale=0.2, num_test_files_class=750, target_size=(500, 500), iterator_mode='arrays',
classifier='tuner', n_threads=1, classes=['class_1', 'class_2', 'class_3'],
project_name='my_project_name')``

The following arguments are acceptable in the :code:`hardy_main()` function:

    * raw_data_path: data_path for the .csv files or images
    * tform_config_path: path for transformation configuration files (.yaml)
    * classifier_config_path: path for hyperparameter search (.yaml)
    * batch_size: batch size for splitting of training and testing of data in machine learning model
    * scale: the scale to which plots are reduce
    * num_test_files_class: The number of test files per class. These files would be reserved for final testing of machine learning model
    * target_size: number of data points in the csv files or dimension of images
    * iterator_mode: if "arrays", the data is fed into machine learning model in array structure. For other values, images files are saved first in .png format and then fed into machine learning model through directory iterators.
    * classifier: tuner or cnn model. Tuner means hyperparameter search while other options execute pre-defined convolutional neural network.
    * n_thread: number of threads used for parallel transformation of data
    * classes: labels or categories in data. If .csv files are used, the label must be present in the filename. If images are used, the images must be contained in respective folders
    * project_name: name for the project. Folder with same name will be created in the raw_data_path containing all the results for the run
    * plot_format: format of the plot to be used for training and testing of data. RGBrgb corresponds to usage of RGB images while any other argument will use cartesian coordinate system.
    * skiprows: Used to skip the metadata contained in the csv files. It must be of same length for all classes.
    * split: The fraction of data used for training and testing of machine learning model. This is different from num_test_files_class since the later one is never fed into machine learning model until the best hyperparameter search is done.
    * seed: the seed used for random-selection of num_test_files_class
    * k_fold: Boolean value indicating whether k-fold validation need to be performed or not
    * k: value indicating how many k-folds need to be performed




