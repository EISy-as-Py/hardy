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
` Guide to write configuration files 
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
.. code::
import hardy.run_hardy as run

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
.. code::
run.hardy_main(raw_data_path, tform_config_path, classifier_config_path, batch_size=64,
scale=0.2, num_test_files_class=750, target_size=(500, 500), iterator_mode='arrays',
classifier='tuner', n_threads=1, classes=['class_1', 'class_2', 'class_3'],
project_name='my_project_name')


.. toctree::
    :maxdepth: 1
    :glob:

    examples/How_to_write_Configuration_files.ipynb
    examples/How_to_make_predictions_using_trained_model.ipynb




