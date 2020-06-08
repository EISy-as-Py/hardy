# import keras
import unittest

from hardy.recognition import tuner


class TestSimulationTools(unittest.TestCase):

    def test_build_param(self):
        param = tuner.build_param('./hardy/recognition/')

        assert isinstance(param, tuner.build_param), \
            'the returned type for param space is incorrect'
        assert isinstance(param.hparam, dict), \
            'the parameter space in config file is not a dictionary'
