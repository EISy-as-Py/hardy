# import keras
import os
import shutil
import unittest
import yaml

from hardy.handling.to_catalogue import learning_set
from hardy.recognition import cnn
from hardy.recognition import tuner

path = './hardy/test/test_image/'
split = 0.25
classes = ['class_1', 'class_2']
batch_size = 1


class TestSimulationTools(unittest.TestCase):

    def test_build_param(self):
        param = tuner.build_param('./hardy/recognition/')

        assert isinstance(param, tuner.build_param), \
            'the returned type for param space is incorrect'
        assert isinstance(param.hparam, dict), \
            'the parameter space in config file is not a dictionary'

    def test_report_generation(self):

        config_path = './hardy/recognition/'
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path=config_path)
        metrics = cnn.evaluate_model(model, val)

        log_dir = './hardy/test/temp_report/'

        tuner.report_generation(model, history, metrics, log_dir,
                                save_model=False, config_path=config_path)

        report_dir = log_dir+'report/'

        report_location = os.listdir(report_dir)

        for item in report_location:
            if item.endswith('.yaml'):
                with open(report_dir+item, 'r') as file:
                    report = yaml.load(file, Loader=yaml.FullLoader)
                    assert isinstance(report, dict),\
                        'The filetype returned in not a dictionary'
        # remove report files after checking they were
        # correctly created
        for item in report_location:
            if item.endswith('.yaml'):
                os.remove(report_dir+item)

    # shutil.rmtree(log_dir)

    def test_run_tuner(self):
        config_path = './hardy/test/'
        tuner.build_param(config_path)

        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        project_name = 'test_project'
        tuner_model = tuner.run_tuner(train, val, project_name='test_project')
        assert tuner_model.oracle.get_space().space[0].name == 'conv_layers',\
            'The name of first layer must be conv_layer'
        assert tuner_model.oracle.get_space().space[1].name == 'filters_0',\
            'First filter must be filters_0'
        assert tuner_model.oracle.get_space().space[2].name == \
            'kernel_size_0', 'First kernel_size must be kernel_size_0'
        assert tuner_model.oracle.get_space().space[3].name == 'activation_0',\
            'First activation layer must be activation_0'
        assert tuner_model.oracle.get_space().space[4].name == 'filters_1',\
            'Second filter must be filters_1'
        assert tuner_model.oracle.get_space().space[5].name == \
            'kernel_size_1', 'Second kernel_size must be kernel_size_1'
        assert tuner_model.oracle.get_space().space[6].name == 'activation_1',\
            'Second activation layer must be activation_1'
        assert tuner_model.oracle.get_space().space[7].name == 'filters_2',\
            'Third filter must be filters_2'
        assert tuner_model.oracle.get_space().space[8].name == \
            'kernel_size_2', 'Third kernel_size must be kernel_size_2'
        assert tuner_model.oracle.get_space().space[9].name == 'activation_2',\
            'Third activation layer must be activation_2'
        assert tuner_model.oracle.get_space().space[10].name == 'pooling',\
            'The pooling layer is absent afer third activation'
        assert tuner_model.oracle.get_space().space[11].name == 'optimizer',\
            'Optimizer should come next to pooling layer'

        # Deleting the log files

        shutil.rmtree('./'+project_name)

        print('Successfully Deleted the log directory created under test')
