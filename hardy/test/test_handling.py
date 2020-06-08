import os
import unittest

# import numpy as np
import pandas as pd

from hardy.handling import handling

data_path = './hardy/test/test_data/'
file_list = os.listdir(data_path)


class TestSimulationTools(unittest.TestCase):

    def test_get_file_list(self):
        """
        Testing Function for getting a list of files that match
            certian file name rules that you may pass
        """
        flist, path = handling.get_file_list(dir_path=data_path,
                                             str_has=['.csv'],
                                             interact=False)

        pass

    def test_check_dir_path(self):
        """
        Testing Function for a does-path-exist check, that also checks for
            files of a certain type inside the path.
        """
        try:
            handling.check_dir_path(data_path)
        except AssertionError:
            pass

    def test_ask_file_list(self):
        """
        Testing Function for Manual TKINTER file box? Not sure what can
            be automated here...
        """
        pass

    def test_cats_from_fnames(self):
        """
        Testing Function for discovering "classification" labels from
            a list of file names, by parsing the end of the files.
        """
        flist, path = handling.get_file_list(dir_path=data_path,
                                             str_has=['.csv'],
                                             interact=False)

        labels_1 = handling.cats_from_fnames(file_list=flist,
                                             path=None,
                                             expect=2,
                                             print_ok=False,
                                             from_serials=False)

        labels_2 = handling.cats_from_fnames(file_list=None,
                                             path=data_path,
                                             expect=2,
                                             print_ok=False,
                                             from_serials=False)
        assert labels_1 == labels_2,\
            "Labels not found equally from path or list? Check fpr Bugs."
        pass

    def test_smart_read_csv(self):
        """
        Testing Function for reading arbitrarilly formed CSV files
            with (for instance) various sized header (so skiprows cannot be
                                                      assumed!)
        """
        file = os.path.join(data_path, file_list[0])
        thedata, rows = handling._smart_read_csv(full_fname=file,
                                                 try_skiprows=6)
        assert type(thedata) is pd.DataFrame, "Error in Smart Data Reader"
        assert type(rows) is int, "Row Count of '{}' is not int?".format(rows)

        pass

    def test_test_df(self):
        """
        Testing Function for ensuring that read dataframes are at least a
        certain size, and has the right data types.
        """
        file = os.path.join(data_path, file_list[0])
        thedata, rows = handling._smart_read_csv(full_fname=file,
                                                 try_skiprows=6)
        testresult = handling.test_df(thedata)
        assert testresult, "Dataframe Test Failed?"
        pass
