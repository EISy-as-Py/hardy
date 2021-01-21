import os
import shutil
import yaml

import unittest
from hardy import run_hardy as run
from hardy.handling import pre_processing as preprocessing
# import pickle
# import numpy as np

path = './hardy/test/test_image/'
data_path = './hardy/test/test_data/'
tform_config_path = data_path + 'tform_config.yaml'
config_path = './hardy/test/'
split = 0.25
classes = ['class_1', 'class_2']
batch_size = 1


class TestSimulationTools(unittest.TestCase):

    def test_hardy_main(self):

        run.hardy_main(
            data_path, tform_config_path, config_path,
            iterator_mode='arrays', classifier='tuner',
            num_test_files_class=1, classes=['noise', 'one'], split=0.5,
            batch_size=1, project_name='test_wrapper')
        output_path = preprocessing.save_to_folder(
                data_path, 'test_wrapper', 'test_1')
        report_dir = output_path+'report/'
        report_location = os.listdir(report_dir)
        for item in report_location:
            if item.endswith('.yaml'):
                with open(report_dir+item, 'r') as file:
                    report = yaml.load(file, Loader=yaml.FullLoader)
                    assert isinstance(report, dict),\
                        'The filetype returned in not a dictionary'
        shutil.rmtree('./hardy/test/test_data/test_wrapper')

        # use k-fold validation
        run.hardy_main(
            data_path, tform_config_path, config_path, k_fold=True, k=2,
            iterator_mode='arrays', classifier='cnn',
            num_test_files_class=1, classes=['noise', 'one'],
            batch_size=1, project_name='test_wrapper')
        output_path = preprocessing.save_to_folder(
                data_path, 'test_wrapper', 'test_1')
        report_dir = output_path+'report/'
        report_location = os.listdir(report_dir)
        for item in report_location:
            if item.endswith('.yaml'):
                with open(report_dir+item, 'r') as file:
                    report = yaml.load(file, Loader=yaml.FullLoader)
                    assert isinstance(report, dict),\
                        'The filetype returned in not a dictionary'
        shutil.rmtree('./hardy/test/test_data/test_wrapper')
        pass

    def test_classifier_wrapper(self):
        num_files = 3
        run_name = 'test_1'
        config_path = './hardy/test/'
        output_path = preprocessing.save_to_folder(
            path, 'test_classifier', run_name)
        test_set_filenames = preprocessing.hold_out_test_set(
            data_path, number_of_files_per_class=num_files)
        run.classifier_wrapper(path, test_set_filenames, run_name,
                               config_path, classifier='cnn',
                               iterator_mode='folder',
                               split=split, classes=classes, image_path=path,
                               project_name='test_classifier')
        report_dir = output_path+'report/'
        report_location = os.listdir(report_dir)
        for item in report_location:
            if item.endswith('.yaml'):
                with open(report_dir+item, 'r') as file:
                    report = yaml.load(file, Loader=yaml.FullLoader)
                    assert isinstance(report, dict),\
                        'The filetype returned in not a dictionary'
        # remove report files after checking they were
        # correctly created
        shutil.rmtree(path+'test_classifier/')
        print('the result folder was correctly deleted after testing')
        pass

    def test_print_time(self):
        duration = [0.75, 50, 200, 4000]
        for i in range(len(duration)):
            t = run.print_time(duration[i])
            assert duration[i] == t, \
                'print statement changed the value of the time '
        pass

    def test_checkrun(self):
        run.checkrun(
            data_path, tform_config_path, config_path,
            iterator_mode='arrays', classifier='tuner',
            classes=['noise', 'one'], split=0.5,
            batch_size=1, project_name='test_wrapper')
