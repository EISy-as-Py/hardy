from hardy.recognition import cnn as cnn
from hardy.recognition import tuner as tuner
from hardy.handling import pre_processing as preprocessing
# from hardy.handling import handling as handling
from hardy.handling import to_catalogue as to_catalogue


def classifier_wrapper(input_path, test_set_filenames, run_name, config_path,
                       image_data=None, classifier='tuner',
                       iterator_mode='arrays', split=0.1,
                       target_size=(80, 80),
                       batch_size=32, image_path=None,
                       classes=['class_1', 'class_2'],
                       project_name='tuner_run', **kwarg):
    '''

    '''
    if iterator_mode == 'arrays':

        assert image_data, 'No image_data list provided'

        test_set_list, learning_set_list = to_catalogue.data_set_split(
            image_data, test_set_filenames)

        training_set, validation_set = to_catalogue.learning_set(
            image_list=learning_set_list, split=split, target_size=target_size,
            iterator_mode='arrays', batch_size=batch_size)

        test_set = to_catalogue.test_set(image_list=test_set_list,
                                         target_size=target_size,
                                         iterator_mode='arrays',
                                         batch_size=batch_size)
    else:

        assert image_path, 'no path to the image folders was provided'

        training_set, validation_set = to_catalogue.learning_set(
            image_path, plit=split, target_size=target_size,
            iterator_mode='from_directory', batch_size=batch_size,
            classes=classes)

        test_set = to_catalogue.test_set(image_path, target_size=target_size,
                                         classes=classes,
                                         iterator_mode='from_directory',
                                         batch_size=batch_size,)

    if classifier == 'tuner':
        # warn search_function, 'no search function provided,
        # using default RandomSearch'
        tuner.build_param(config_path)
        tuned_model = tuner.run_tuner(training_set, validation_set,
                                      project_name=project_name +
                                      run_name)
        model, history, metrics = tuner.best_model(tuned_model, training_set,
                                                   validation_set, test_set)
        output_path = preprocessing.save_to_folder(input_path, project_name,
                                                   run_name)
        conf_matrix, report = cnn.report_on_metrics(model, test_set) 
        tuner.report_generation(model, history, metrics, output_path,
                                tuner=tuned_model, save_model=True)
    else:
        model, history = cnn.build_model(training_set, validation_set,
                                         config_path=config_path)
        metrics = cnn.evaluate_model(model, test_set)

        output_path = preprocessing.save_to_folder(input_path, project_name,
                                                   run_name)
        conf_matrix, report = cnn.report_on_metrics(model, test_set)
        tuner.report_generation(model, history, metrics, output_path,
                                tuner=None, save_model=True,
                                config_path=config_path)

    return
