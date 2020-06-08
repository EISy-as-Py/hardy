# import keras
import os
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
    # shutil.rmtree(log_dir)
