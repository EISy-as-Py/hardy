import numpy as np
import unittest


from hardy.handling.visualization import (normalize, normalize_image, rgb_plot,
                                          orthogonal_images_add,
                                          orthogonal_images_mlt)

#####################################################################
# Define arrays for testing the funcitons in visualization
array_to_normalize = np.linspace(5, 25, 100)
image_to_normalize[:, :, 0] = array_to_normalize
image_to_normalize[:, :, 1] = array_to_normalize*2 + 1
image_to_normalize[:, :, 2] = np.sqrt(array_to_normalize) + 2


class TestSimulationTools(unittest.TestCase):

    def test_normalize(self):
        normalized_array = visualization.normalize(array_to_normalized)
        assert isinstance(array_to_normalized, (pd.core.series.Series,
                          np.ndarray)), \
            'the input of the normalization function should be an ' +\
            'array or a pandas dataframe'
        assert 0 <= min(normalized_array), \
            'the lower limit of the normalized array should be zero'
        np.testing.assert_almost_equal(max(normalized_array),
                                       1, decimal=18, err_msg='the ormalized \
                                       array is nto correclty computed.')

    def test_normalize_image(self):
        normalized_image = visualization.normalize_image(image_to_normalized)
        assert isinstance(image_to_normalized, np.ndarray),\
            'the input of the normalization function should be an array'
        assert np.shape(normalized_image)[0] == np.shape(normalized_image)[1],\
            'the normalized image is not square'
        assert 0 <= min(normalized_image[:, :, 0]), \
            'the lower limit of the normalized image should be zero'
        np.testing.assert_almost_equal(max(normalized_image[:, :, 0]),
                                       1, decimal=18, err_msg='the ormalized \
                                       array is nto correclty computed.')

    def test_rgb_plot(self):
        rgb_plot_array = visualization.rgb_plot(red_array=array_to_normalize)
        assert isinstance(array_to_normalized, (pd.core.series.Series,
                          np.ndarray)), \
            'the input of the plotting function should be an ' +\
            'array or a pandas dataframe'
        assert isinstance(rgb_plot_array, np.ndarray), 'the resulting image' +\
            'should be a numpy array'

    def test_orthogonal_images_add(self):
        rgb_plot_x = visualization.rgb_plot(red_array=array_to_normalize)
        rgb_plot_y = visualization.rgb_plot(blue_array=(1/array_to_normalize))
        rgb_plot_array = visualization.orthogonal_images_add(rgb_plot_x,
                                                             rgb_plot_y)
        assert isinstance(rgb_plot_x, np.ndarray), \
            'the input of the plotting function should be an ' +\
            'array or a pandas dataframe'
        assert isinstance(rgb_plot_array, np.ndarray), 'the resulting image' +\
            'should be a numpy array'

    def test_orthogonal_images_add(self):
        rgb_plot_x = visualization.rgb_plot(red_array=array_to_normalize)
        rgb_plot_y = visualization.rgb_plot(blue_array=(1/array_to_normalize))
        rgb_plot_array = visualization.orthogonal_images_mlt(rgb_plot_x,
                                                             rgb_plot_y)
        assert isinstance(rgb_plot_x, np.ndarray), \
            'the input of the plotting function should be an ' +\
            'array or a pandas dataframe'
        assert isinstance(rgb_plot_array, np.ndarray), 'the resulting image' +\
            'should be a numpy array'
