import plotly
import shutil
import unittest

import hardy.data_reporting.reporting as reporting

import pandas as pd

from hardy import run_hardy as run
from hardy.handling import pre_processing as preprocessing
from hardy.handling import to_catalogue as catalogue
from hardy.recognition import cnn

# from hardy.handling import pre_processing as preprocessing

path = './hardy/test/test_image/'
data_path = './hardy/test/test_data/'
tform_config_path = data_path + 'tform_config.yaml'
config_path = './hardy/test/'
split = 0.25


class TestSimulationTools(unittest.TestCase):

    def test_summary_report_plots(self):

        run.hardy_multi_transform(
            data_path, tform_config_path, config_path,
            iterator_mode='arrays',
            num_test_files_class=1, classes=['noise', 'one'], split=0.25,
            classifier='tuner', batch_size=1, project_name='test_wrapper1')
        report_path = './hardy/test/test_data/test_wrapper1/'
        fig1, fig2 = reporting.summary_report_plots(
            report_path)

        assert isinstance(fig1, plotly.graph_objs._figure.Figure),\
            'The returned figure is not a plotly object'
        assert isinstance(fig2, plotly.graph_objs._figure.Figure),\
            'The returned figure is not a plotly object'

        # shutil.rmtree('./hardy/test/temp_report')
        # shutil.rmtree('./test_run')

    def test_summary_report_tables(self):
        report_path = './hardy/test/test_data/test_wrapper1/'

        summary, tform_rank = reporting.summary_report_tables(
            report_path)

        assert isinstance(summary, pd.DataFrame),\
            'The returned table is not a dataframe'
        assert isinstance(tform_rank, pd.DataFrame),\
            'The returned table is not a dataframe'

        shutil.rmtree('./hardy/test/test_data/test_wrapper1')
        # shutil.rmtree('./test_run')

    def test_model_analysis(self):

        num_files = 1
        data_tups = catalogue._data_tuples_from_fnames(input_path=data_path)

        plot_tups = catalogue.rgb_list(data_tups)

        test_set_filenames = preprocessing.hold_out_test_set(
            data_path, number_of_files_per_class=num_files)

        test_set_list, learning_set_list = catalogue.data_set_split(
            plot_tups, test_set_filenames)
        train, val = catalogue.learning_set(image_list=learning_set_list,
                                            split=split,
                                            classes=['noise', 'one'],
                                            iterator_mode='arrays')
        testing_set = catalogue.test_set(image_list=test_set_list,
                                         classes=['noise', 'one'],
                                         iterator_mode='arrays')
        model, history = cnn.build_model(train, val,
                                         config_path='./hardy/test/')

        result = reporting.model_analysis(model, testing_set, test_set_list)

        assert isinstance(result, pd.DataFrame)
