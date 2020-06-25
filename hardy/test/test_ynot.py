import unittest
import pandas as pd

from hardy.ynot import ynot as y


class TestSimulationTools(unittest.TestCase):

    def test_generate_linear(self):

        results = y.generate_linear(length=64, xrange=[0, 1], m=1.0, b=1.0)

        assert isinstance(results, pd.DataFrame), \
            "the data generated should be stored in a dataframe"

    def test_generate_sin(self):

        results = y.generate_sin(length=64, xrange=[0, 1], A=1.0,
                                 f=1.0, theta=0.0)

        assert isinstance(results, pd.DataFrame), \
            "the data generated should be stored in a dataframe"

    def test_generate_log(self):

        results = y.generate_log(length=64, xrange=[0, 1],
                                 A=1.0, y0=0, x0=-0.5)

        assert isinstance(results, pd.DataFrame), \
            "the data generated should be stored in a dataframe"
