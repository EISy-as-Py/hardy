import unittest

# import numpy as np
import os
import pandas as pd

# import hardy.arbitrage.arbitrage as arbitrage
from hardy.arbitrage import arbitrage


data_path = './test_data/'
tform_config_path = './test_data/test_tform_config.yaml'

assert os.path.exists(data_path), \
    "Did not find Test Data Files at {}".format(data_path)

sample_data = os.listdir(data_path)  # List of file names in data_path

for file in sample_data:
    fname = file[:-4]  # File name is the file without extension
    raw_df = pd.read_csv(os.path.join(data_path, file), skiprows=6)
    label = fname[-5:] # classification

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
        Testing Function for
        """
        pass

    def test_tform_tuples(self):
        """
        Testing Function for the transform Wrapping function,
            which should take the list-of-tuples and apply the transform
            to each.
        """
        pass


# =============================================================================
# """
# TESTING ZONE
# """
#
# a=time.perf_counter()
# test_dir = '../local_data/2020-4-21_0000'
# test_tform_list = (
#      (0, "1d_exp", 0),
#      (1, "1d_none", 1),
#      (2, "1d_cumsum", 1),
#      )
#
# transformed_data, save_path = setup_tform_files(test_dir)
# test_tform_data = load_and_transform_data(transformed_data, test_tform_list,
#                                           save_path=save_path)
#
# print("Time was : {} seconds".format(round(time.perf_counter()-a,2)))
# """
# To-Do now:
#    -make some sort of iterable to scan the data and determine GLOBAL features
#    (like max and min, to deterimine what transforms are OK for each dataset)
#    -Use this function to create a LIST of Transform_list possibilities, and
#    iterate over that list (as we will be able to direct)
#
# """
# =============================================================================
