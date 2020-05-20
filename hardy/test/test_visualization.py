import unittest

import numpy as np
import pandas as pd

# import hardy.handling.visualization as visualization
from hardy.handling.visualization import (normalize, normalize_image, rgb_plot,
                                          orthogonal_images_add,
                                          orthogonal_images_mlt)

#####################################################################
# Define arrays for testing the funcitons in visualization
array_to_normalize = np.linspace(5, 25, 100)

# Needed to Pre-Initialize the Image ndarray.
n = len(array_to_normalize)
image_to_normalize = np.ndarray([n, n, 3])

image_to_normalize[:, :, 0] = array_to_normalize
image_to_normalize[:, :, 1] = array_to_normalize*2 + 1
image_to_normalize[:, :, 2] = np.sqrt(array_to_normalize) + 2


class TestSimulationTools(unittest.TestCase):

    def test_normalize(self):
        normalized_array = normalize(array_to_normalize)
        assert isinstance(array_to_normalize, (pd.core.series.Series,
                          np.ndarray)), \
            'the input of the normalization function should be an ' +\
            'array or a pandas dataframe'
        assert 0 <= min(normalized_array), \
            'the lower limit of the normalized array should be zero'
        np.testing.assert_almost_equal(max(normalized_array),
                                       1, decimal=18, err_msg='the ormalized \
                                       array is nto correclty computed.')

    def test_normalize_image(self):
        normalized_image = normalize_image(image_to_normalize)
        assert isinstance(image_to_normalize, np.ndarray),\
            'the input of the normalization function should be an array'
        assert np.shape(normalized_image)[0] == np.shape(normalized_image)[1],\
            'the normalized image is not square'
        assert 0 <= min(normalized_image[:, :, 0]), \
            'the lower limit of the normalized image should be zero'
        np.testing.assert_almost_equal(max(normalized_image[:, :, 0]),
                                       1, decimal=18, err_msg='the ormalized \
                                       array is nto correclty computed.')

    def test_rgb_plot(self):
        rgb_plot_array = rgb_plot(red_array=array_to_normalize)
        assert isinstance(array_to_normalize, (pd.core.series.Series,
                          np.ndarray)), \
            'the input of the plotting function should be an ' +\
            'array or a pandas dataframe'
        assert isinstance(rgb_plot_array, np.ndarray), \
            'the resulting image should be a numpy array'

    def test_orthogonal_images_add(self):
        rgb_plot_x = rgb_plot(red_array=array_to_normalize)
        rgb_plot_y = rgb_plot(blue_array=(1/array_to_normalize))
        rgb_plot_array = orthogonal_images_add(rgb_plot_x, rgb_plot_y)
        assert isinstance(rgb_plot_x, np.ndarray), \
            'the input of the plotting function should be an ' +\
            'array or a pandas dataframe'
        assert isinstance(rgb_plot_array, np.ndarray), \
            'the resulting image should be a numpy array'

    def test_orthogonal_images_mlt(self):
        rgb_plot_x = rgb_plot(red_array=array_to_normalize)
        rgb_plot_y = rgb_plot(blue_array=(1/array_to_normalize))
        rgb_plot_array = orthogonal_images_mlt(rgb_plot_x, rgb_plot_y)
        assert isinstance(rgb_plot_x, np.ndarray), \
            'the input of the plotting function should be an ' +\
            'array or a pandas dataframe'
        assert isinstance(rgb_plot_array, np.ndarray), \
            'the resulting image should be a numpy array'
