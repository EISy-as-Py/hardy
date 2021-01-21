import os
import unittest

import pandas as pd

from hardy.arbitrage import arbitrage
from hardy.handling import handling

data_path = './hardy/test/test_data/'
tform_config_path = './hardy/test/test_data/test_tform_config.yaml'


assert os.path.exists(data_path), \
    "Did not find Test Data Files at {}".format(data_path)

sample_data = os.listdir(data_path)  # List of file names in data_path
sample_tuples = []

for file in sample_data:
    if '.csv' in file:
        fname = file[:-4]  # File name is the file without extension
        raw_df = handling.read_csv(os.path.join(data_path, file),
                                   skiprows=0)
        raw_df = pd.read_csv(os.path.join(data_path, file), skiprows=0)
        label = fname[-5:]
        # ^ Label is the last part of the fname (just testing)
        the_tuple = (fname, raw_df, label)
        sample_tuples.append(the_tuple)
    else:
        pass

raw_df = sample_tuples[0][1]
# Use first (0th) tuple, DF in position 1.

tform_example = [[0, 'raw', 0],
                 [1, 'nlog', 1],
                 [5, 'nlog', 2]]


class TestSimulationTools(unittest.TestCase):

    def test_import_tform_config(self):
        """
        Testing Function for importing the Configuration file.
        """
        assert os.path.exists(tform_config_path), \
            "Did not find Tform Config file at {}".format(tform_config_path)
        try:
            arbitrage.import_tform_config(tform_config_path, raw_df=raw_df)
        except AssertionError:
            pass

    def test_apply_tform(self):
        """
        Testing Function for applying a transform to each dataframe
        """
        try:
            arbitrage.apply_tform(raw_df, tform_example)
        except AssertionError:
            pass

    def test_tform_tuples(self):
        """
        Testing Function for the transform Wrapping function,
            which should take the list-of-tuples and apply the transform
            to each.
        """
        try:
            arbitrage.tform_tuples(sample_tuples, tform_example)
        except AssertionError:
            pass
