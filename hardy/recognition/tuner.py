import datetime
import os
import yaml

import kerastuner as kt
import tensorflow as tf


class build_param():
    def __init__(self, config_path):
        with open(config_path + 'tuner_config.yaml', 'r') as file:
            self.hparam = yaml.load(file, Loader=yaml.FullLoader)
        global tuner_parameters
        tuner_parameters = self.hparam


def build_tuner_model(hp):
    '''
    Functions that builds a convolutional keras model with
    tunable hyperparameters


    Parameters
    ----------
    hp: keras tuner class
        A class that is used to define the parameter search space

    Returns
    -------
    model: Keras sequential model
           The trained convolutional neural network
    '''
    ###################################
    # loading the configuration file for tuner

    param = tuner_parameters

    ####################################
    # Defining input size
    # need to put input shape in the config file
    input = (param['input_shape'][0], param['input_shape'][0],
             param['input_shape'][1])
    inputs = tf.keras.Input(shape=input)
    x = inputs

    ####################################
    # extracting parameters from the parameters file
    # and feeding in the tuner
    kernel = getattr(
        hp, param['kernel_size'][0])(
        'kernel_size', values=param['kernel_size'][1]['values']),
    kernel_size = (kernel[0], kernel[0])

    filter = getattr(
        hp, param['filters'][0])(
            'filters', min(param['filters'][1]['values']),
            max(param['filters'][1]['values']), step=4, default=8)

    for i in range(hp.Int('conv_layers', 1, max(param['layers']),
                          default=3)):
        x = tf.keras.layers.Conv2D(
            filters=filter*(i+1),
            kernel_size=kernel_size,
            activation=getattr(hp, param['activation'][0])
            ('activation_' + str(i+1), values=param['activation'][1]['values']
             ), padding='same')(x)

    if getattr(hp,
               param['pooling'][0])('pooling',
                                    values=param['pooling'][1]['values'])\
            == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(
        param['num_classes'][0], activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # adding in the optimizer
    optimizer = getattr(hp, param['optimizer'][0])('optimizer',
                                                   values=param['optimizer']
                                                   [1]['values'])

    # compiling neural network model
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def best_model(tuner, training_set, validation_set, test_set):
    '''
    Function that takes the tuner and builds up the model on the basis on best
    hyperparameters in the tuner

    Parameters
    ----------

    tuner: keras tuner
           tuner generated by specifications from tuner_build_model function
    training_set: keras pointer
                  training set data generated through flow from directory
    validation_set: keras pointer
                    validation set data generated through flow from directory
    test_set: keras point
              test_set data generated through flow from directory. Used for
              cross validation of model
    epochs: int
            the number of times model is executed to be trained over
            training set & validation set

    Returns
    -------

    model: keras model
           model built up using the best hyperparameters in the tuner
    history: dict
             dictionary containing result from fitting model oveer training
             and validation set
    metrics: np.float64
             np array containing loss and accuracy for cross-validation of data

    '''
    param = tuner_parameters

    best_hp = tuner.get_best_hyperparameters()[0]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=param['patience'
                                                                     ][0])
    # best_hp_values = best_hp.values

    model = tuner.hypermodel.build(best_hp)

    history = model.fit(training_set, epochs=param['epochs'][0], verbose=0,
                        validation_data=validation_set,
                        callbacks=[early_stopping])

    metrics = model.evaluate(test_set, verbose=0)

    return model, history, metrics


def run_tuner(training_set, validation_set, project_name='untransformed'):
    '''
    Function that runs the tuner using training set, validation set and
    hyperparameters defined in the config file

    Parameters
    ----------
    training_set: keras pointer
                  training set data generated through keras flow from
                  directory function
    validation_set: keras pointer
                    validation set data generated through keras flow from
                    directory function
    project_name: str
                  name to use for the log files of the tuner run

    Returns
    -------
    tuner: keras tuner
    '''
    param = tuner_parameters

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=param['patience'
                                                                     ][0])

    if param['search_function'][0] == 'BayesianOptimization':
        tuner = getattr(kt.tuners, param['search_function'][0]
                        )(build_tuner_model, objective='val_accuracy',
                          max_trials=param['max_trials'][0],
                          # alpha=param['alpha'][0],
                          # beta=param['beta'][0],
                          executions_per_trial=param['exec_per_trial'][0],
                          project_name=project_name)

    else:
        tuner = getattr(kt.tuners, param['search_function'][0]
                        )(build_tuner_model, objective='val_accuracy',
                          max_trials=param['max_trials'][0],
                          executions_per_trial=param['exec_per_trial'][0],
                          project_name=project_name)

    tuner.search(training_set, epochs=param['epochs'][0],
                 validation_data=validation_set,
                 verbose=2, callbacks=[early_stopping])

    return tuner


def report_generation(model, history, metrics, log_dir,
                      tuner=None, save_model=True, config_path=None,
                      k_fold=False, k=None):
    '''
    Function that generates the report based on tuner search
    and hyperparameters

    Parameters
    ----------
    tuner: keras tuner
           tuner generated by the run_tuner function
    model: keras model
           model built up using the best hyperparameters in the tuner
           generated by the tuner 'best_model' function
    history: history: dict
             dictionary containing result from fitting model over training
             and validation set generated by best_model function
    metrics: np.float64
             np array containing loss and accuracy for cross-validation of
             data generated by best_model function
    log_dir: str
             string representing the location where the report needs to be
             stored
    save_model: bool
                If true saves the model with best hyperparameters in the
                defined location
    config_path: str
                 location of configuration file for the convolutional neural
                 network

    Returns
    -------

    .yaml file containing the hyperparameters, performance and history
    of the trained CNN
    '''

    if tuner is not None:
        best_hp = tuner.get_best_hyperparameters()[0].values
    else:
        assert (config_path), "Please,Provide the config path"
        with open(config_path + 'cnn_config.yaml', 'r') as file:
            best_hp = yaml.load(file, Loader=yaml.FullLoader)

    if save_model:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_location = log_dir+'best_model'
        model.save(model_location+'.h5')
        model_location_dict = {'model_location': model_location}
    else:
        model_location = 'None'
        model_location_dict = {'model_location': model_location}

    metrics_accuracy = history.__dict__['history']

    metrics_accuracy_feed = {}
    for key, value in metrics_accuracy.items():
        metrics_accuracy_feed.update({key: [float(item) for item in value]})

    validation_metrics_dict = {'test_loss': float(metrics[0]),
                               'test_accuracy': float(metrics[1])}

    report_location = log_dir+'/report/'
    if not os.path.exists(report_location):
        os.makedirs(report_location)

    with open(report_location+datetime.datetime.now().strftime(
             "%y%m%d_%H%M") + ".yaml", 'w') as yaml_file:
        yaml.dump(best_hp, yaml_file)
        yaml.dump(metrics_accuracy_feed, yaml_file)
        yaml.dump(validation_metrics_dict, yaml_file)
        yaml.dump(model_location_dict, yaml_file)
        if k_fold:
            k_val = {'k_folds': k}
            yaml.dump(k_val, yaml_file)
        yaml_file.close()
    return
