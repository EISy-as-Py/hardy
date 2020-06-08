import unittest

# import numpy as np
import os
# import pandas as pd

# import hardy.arbitrage.arbitrage as arbitrage
from hardy.arbitrage import arbitrage
from hardy.handling import handling

data_path = './hardy/test/test_data/'
tform_config_path = './hardy/test/test_data/test_tform_config.yaml'

assert os.path.exists(data_path), \
    "Did not find Test Data Files at {}".format(data_path)

sample_data = os.listdir(data_path)  # List of file names in data_path
sample_tuples = []

for file in sample_data:
    fname = file[:-4]  # File name is the file without extension
    raw_df = handling._smart_read_csv(os.path.join(data_path, file),
                                      try_skiprows=6)
    label = fname[-5:]  # Label is the last part of the fname (just testing)
    the_tuple = (fname, raw_df, label)
    sample_tuples.append(the_tuple)

raw_df = sample_data[0][1]

tform_example = [[0, '1d_raw', 0],
                 [1, '1d_log', 1],
                 [5, '1d_log', 2]]


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


# =============================================================================
# """
# TESTING ZONE
# """
#
# To-Do now:
#    -make some sort of iterable to scan the data and determine GLOBAL features
#    (like max and min, to deterimine what transforms are OK for each dataset)
#    -Use this function to create a LIST of Transform_list possibilities, and
#    iterate over that list (as we will be able to direct)
#
# """
# =============================================================================
