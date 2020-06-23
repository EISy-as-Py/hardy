__Example script to run the HARDy Package__

##### 1. import the package

<div style='padding-left:30px; padding-top:30px'><code>import hardy.run_hardy as run</code></div>


##### 2. Provide the path to the :

_Note: the configuration path shown are the default path. These can be modified if the configuration files used are stored ina  different folder_

* The raw .csv data

<div style='padding-left:80px'><code>raw_data_path = <'path/to/raw/data/'></code></div>


* The configuration file containing the transformations


<div style='padding-left:80px'><code>tform_config_path = <'./hardy/arbitrage/tform_config.yaml' ></code></div>

* The configuration file for the classifier


<div style='padding-left:80px'><code>classifier_config_path = <'./hardy/recognition/' ></code></div>



##### 3. Execute the function to run the code


<div style='padding-left:30px; padding-top:30px'><code>run.hardy_multi_transform(raw_data_path, tform_config_path, classifier_config_path, batch_size=64, scale=0.2,
    num_test_files_class=750, target_size=(500, 500), iterator_mode='arrays',classifier='tuner',
    classes=['class_1', 'class_2', 'class_3'], project_name='my_project_name')</code></div>


_Note: To see the detailed descriptions of the arguments and defaults values, look at the <a href=https://github.com/EISy-as-Py/hardy/blob/master/hardy/run_hardy.py>run_hardy.py docstrings </a>_ 
