import unittest

import numpy as np
import pandas as pd

from hardy.arbitrage import transformations as tforms


class TestSimulationTools(unittest.TestCase):

    def test_transform_1d_exp(self):
        test_array = [1, 2, 3, 4, 5]
        result = tforms.transform_1d_exp(test_array)

        assert len(test_array) == len(result), "The returned size for\
                array is invalid"

    def test_transform_1d_reciprocal(self):
        test_array = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = tforms.transform_1d_reciprocal(test_array)

        assert len(test_array) == len(result), "The returned size for\
                array is invalid"

    def test_transform_1d_cumsum(self):
        test_array = [1, 2, 3, 4, 5, 6]

        result = tforms.transform_1d_cumsum(test_array)

        assert len(test_array) == len(result), "The returned size for\
                array is invalid"
        assert result[-1] == 21, "The output return is not correct"

    def test_transform_1d_derivative(self):
        test_array = [1, 2, 3, 4, 5, 6]

        result = tforms.transform_1d_derivative(test_array)

        assert len(test_array) == len(result), "The returned size for\
                array is invalid"
        assert sum(result) == len(test_array), "The returned sum is\
                not correct"

        result = tforms.transform_1d_derivative(test_array,
                                                spacing=1)

        assert len(test_array) == len(result), "The returned size for\
                array is invalid"
        assert sum(result) == len(test_array), "The returned sum is\
                not correct"

    def test_transform_array_multiplication(self):
        test_array_x = [2, 2, 3, 3, 4, 4]
        test_array_y = [4, 2, 3, 3, 4, 4]

        result = tforms.transform_array_multiplication(test_array_x,
                                                       test_array_y)
        assert len(test_array_x) == len(result), "The returned size for\
                array is invalid"
        assert test_array_x[0]*test_array_y[0] == result[0], "The returned\
                multiplication output is not correct"

        result = tforms.transform_array_multiplication(test_array_x)
        assert len(test_array_x) == len(result), "The returned size for\
                array is invalid"
        assert test_array_x[0]*test_array_x[0] == result[0], "The returned\
                multiplication output is not correct"

    def test_tform_1d_cwt(self):
        """
        Testing Package for a 1-dimensional Continuous-Wavelet Transform
        This is a type of Fourier transform, to demonstrate chaning frequency
        over time. See Here, Scipy:
            https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.signal.cwt.html#scipy.signal.cwt
        As with other transforms starting with 1D data
            (pass X or Y, or as 0 or 1), intake a 1D array and output a
            2D np array that is a square. HOWEVER unlike most 1D transforms,
            this will have actual 2D shape.
        The X-axis (regardless of X or Y data transform) will have length units
            (recording position in file) but the Second Axis will have
            Frequency units: Highest frequency at the top(?) and lowest
            at the bottom, where top is a wavelet of minimum size
            (2 datapoints?) and the bottom is maximum wavelet size (all data?)
            ... I need to test this... Just use the sp.

        Should we check the fidelity of a Reverse-Transform??

        """
        # Starting with a linear x from 0 to 10,
        x_linear = np.linspace(0, 10, 500)
        # generate a complex wave function:
        #   In this case, amplitude 1 with frequency of 2pi/10, plus
        #   Amplitude 1/10 with frequency of 10pi.
        y_test = np.sin(2 * np.pi * 0.1 * x_linear) + \
            0.1 * np.sin(2 * np.pi * 5 * x_linear)

        test_df = pd.DataFrame(data={"Xlinear": x_linear, "Ytest": y_test})

        # Valid Test: Pass Test_df and the Y-axis transform to get output df
        # of the Y-test data.

        result_1 = tforms.transform_1d_cwt(test_df, 1)
        result_y = tforms.transform_1d_cwt(test_df, "y")
        assert np.allclose(result_1, result_y), "Not accepting y as 1 input"

    def test_2d_derivative(self):

        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [9, 10, 11, 12, 13, 14, 15, 16]

        check_result = [1, 1, 1, 1, 1, 1, 1]

        slope = tforms.transform_2d_derivative(x, y)

        assert np.allclose(slope, check_result), "The returned array for slope,\
        contains incorrect elements"
