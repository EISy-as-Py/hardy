# import keras
import os
import shutil
import unittest
import yaml

from hardy.handling.to_catalogue import learning_set
from hardy.recognition import cnn
from hardy.recognition import tuner

path = os.path.join('.', 'hardy', 'test', 'test_image')
split = 0.25
classes = ['class_1', 'class_2']
batch_size = 1


class TestSimulationTools(unittest.TestCase):

    def test_build_param(self):
        param = tuner.build_param(os.path.join('.', 'hardy', 'recognition'))

        assert isinstance(param, tuner.build_param), \
            'the returned type for param space is incorrect'
        assert isinstance(param.hparam, dict), \
            'the parameter space in config file is not a dictionary'

    def test_report_generation(self):

        config_path = os.path.join('.', 'hardy', 'test')
        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        model, history = cnn.build_model(train, val,
                                         config_path=config_path)
        metrics = cnn.evaluate_model(model, val)

        log_dir = os.path.join('.', 'hardy', 'test', 'temp_report')

        tuner.report_generation(model, history, metrics, log_dir,
                                save_model=False, config_path=config_path)

        report_dir = os.path.join(log_dir, 'report')

        report_location = os.listdir(report_dir)

        for item in report_location:
            if item.endswith('.yaml'):
                with open(os.path.join(report_dir, item), 'r') as file:
                    report = yaml.load(file, Loader=yaml.FullLoader)
                    assert isinstance(report, dict),\
                        'The filetype returned in not a dictionary'
        # remove report files after checking they were
        # correctly created
        # for item in report_location:
        #     if item.endswith('.yaml'):
        #         os.remove(report_dir+item)

        shutil.rmtree(log_dir)

    def test_run_tuner(self):
        config_path = os.path.join('.', 'hardy', 'test')
        tuner.build_param(config_path)

        train, val = learning_set(path, split=split, classes=classes,
                                  iterator_mode=None)
        project_name = 'test_project'
        tuner_model = tuner.run_tuner(train, val, project_name='test_project')
        assert tuner_model.oracle.get_space().space[0].name == \
            'kernel_size', 'The first entry should be the kernel size'
        assert tuner_model.oracle.get_space().space[1].name == 'filters',\
            'The name of first layer must be filters'
        assert tuner_model.oracle.get_space().space[2].name == 'conv_layers',\
            'First filter must be conv_layers'

        # Deleting the log files

        shutil.rmtree(os.path.join('.', project_name))

        print('Successfully Deleted the log directory created under test')
        # Generate test for BayesianOptimization search function
        # config_path = './hardy/test/test_data/'
        # tuner.build_param(config_path)
        # tuner_model = tuner.run_tuner(train, val, project_name='test_project'
        # )
        # assert tuner_model.oracle.get_space().space[0].name == \
        #     'kernel_size', 'The first entry should be the kernel size'
        # assert tuner_model.oracle.get_space().space[1].name == 'filters',\
        #     'The name of first layer must be filters'
        # assert tuner_model.oracle.get_space().space[2].name == 'conv_layers',
        # \
        #     'First filter must be conv_layers'

        # shutil.rmtree('./'+project_name)

        print('Successfully Deleted the log directory created under test')
