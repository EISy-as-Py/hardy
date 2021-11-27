import numpy as np
import pandas as pd
from scipy import signal


def raw(raw_array):
    ''' Function that provides returns data as it is

    Parameters
    ----------
    raw_array: numpy.array
               array representing data values

    Returns
    -------
    raw_array: numpy.array
               array representing data values
    
    Notes
    -----
    .. math::
        Z = Z

    '''
    # Placeholder, to perform "no transform" and use the raw data in that
    # column
    return raw_array


def exp(raw_array):
    '''Function that returns the exponent of individual elements in
    the array

    Parameters
    ----------
    raw_array: numpy.array
               array representing data values
    
    Returns
    -------
    exp_array: numpy.array
               array representing the exponentials of data values
    
    Notes
    -----
    .. math::
        Z = \\exp{Z}

    '''
    # import numpy as np
    # Simple transform, returning the exponential value of each number
    return np.exp(raw_array)


def nlog(raw_array):
    '''The function that outputs the natural log of input array

    Parameters
    ----------
    raw_array: Input numpy array

    Returns
    -------
    log_array: np.ndarray
               natural log values of each element in the input array
    '''

    # NOTE: All Elements in array MUST be Positive!?
    #       IF Not, option to normalize first??
    assert min(raw_array) > 0, "Log will not accept negative values!"
    log_array = np.log(raw_array)
    return log_array


def log10(raw_array):
    '''The function that outputs the natural log of input array

    Parameters
    ----------
    raw_array: Input numpy array

    Returns
    -------
    log_array: np.ndarray
               natural log values of each element in the input array
    '''

    # NOTE: All Elements in array MUST be Positive!?
    #       IF Not, option to normalize first??
    assert min(raw_array) > 0, "Log will not accept negative values!"
    log_array = np.log10(raw_array)
    return log_array


def reciprocal(raw_array):
    '''The function the outputs the reciprocal of input array

    Parameters
    ----------
    raw_array: Input numpy array

    Returns
    -------
    reciprocal_array: np.ndarray
               reciprocal values of each element in the input array
    '''
    reciprocal_array = np.reciprocal(raw_array)
    return reciprocal_array


def cumsum(raw_array):
    '''The function return the cumulative sum of input array

    Parameters
    ----------
    raw_array: Input numpy array

    Returns
    -------
    cumsum _array: np.ndarray
               cumulative sum of values in the input array
    '''
    cumsum_array = np.cumsum(raw_array)
    return cumsum_array


def derivative_1d(raw_array, spacing=0):
    ''' Function that outputs the gradient of 1-D array using
    numpy.gradient function

    Parameters
    ----------
    raw_array: numpy array
    spacing: int representing the spacing between each datapoint

    Returns
    -------
    derivative_array: np.ndarray
                      array representing gradient at each datapoint
    '''

    if spacing == 0:
        spacing = np.arange(np.size(raw_array))
    else:
        spacing = spacing

    derivative_array = np.gradient(raw_array, spacing)

    return derivative_array


def derivative_2d(x, y, meta_data=None):
    """Function that outputs the slope between x and y data

    Parameters
    ----------
    x: numpy.array
       array representing values on x-axis
    y: numpy.array
       array representing values on y-axis

    Returns
    -------
    slope_array: numpy.array
                 array representing the slope between x and y
    """

    diff_x = np.diff(x)
    diff_y = np.diff(y)

    slope_array = diff_y/diff_x

    slope_array = np.concatenate((slope_array, np.array([0])))

    return slope_array


def cwt_1d(raw_df, xy=0):
    """
    Transform to execute a "Continuous Wavelet Transform" on a 1d data array
    pass it a raw XY data and tell it which column to use for the transform.
    See Documentaion on CWT transform:
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.signal.cwt.html#scipy.signal.cwt
    Note: I need to do testing to understand the in/outputs here...
    Plan is to simply hard-code a certain type of Wavelet to use... and
    Output Data may not be able to be square... In that case, we will
    discuss how to integrate this result with the compression of the data.

    Parameters
    ----------
    raw_df: pandas.DataFrame or 1D array (Mx2 or Mx1)
                the raw data which is to be transformed.
    xy:     boolean, or string 'x', or 'y'
                information on which dataframe column to transform.
                ignored if an 1D array is passed instead.
    w_method: string or boolean?
                input instructions guiding how to choose wavelet sizes.
                default should be linear, with options for log- or exponential?
                (Will have to experiment with data to discover best option)

    Returns
    ----------
    cwt_matrix: np.ndarray (MxM)
                Square M-by-M matrix of the wavelet transform data
                (Not yet compressed to plottable 0-1 data)

    """
    if type(raw_df) is pd.DataFrame:
        # Optional User-input, accept "y" or "Y" strings as 1, etc
        if xy == "x" or xy == "X":
            xy = 0
        elif xy == "y" or xy == "Y":
            xy = 1

        if xy == 0:
            data = raw_df[raw_df.keys()[0]]
        elif xy == 1:
            data = raw_df[raw_df.keys()[1]]

    elif type(raw_df) is np.ndarray:
        if raw_df.size == raw_df.shape[0]:
            data = raw_df
        else:
            data = raw_df[0]
        assert len(data) > 10, "NDarray is too small!"
        assert data.size == data.shape[0], "NDarray is Multi-dimensional"

    else:
        # If not DataFrame or NDarray... What is it? Pd.Series?
        # Will continue to creat tests to handle datatypes...
        data = raw_df
        assert len(data) > 10, "Something wrong with data entry." + \
            "Needs Dataframe or 1-Dimensional Data Array!"

    data_n = len(data)
    widths = np.arange(1, data_n, 1)
    cwt_matrix = signal.cwt(data, signal.ricker, widths)
    # Optional different Signal to compare with: "Morlet2" but not working?)
    # cwt_matrix = signal.cwt(data, signal.morlet, widths)

    return cwt_matrix


def power(x, y='None', meta_data=None):
    ''' Function that multiplies two arrays x^m & y^n, element
    by element. If y is None, it return x*x

    Parameters
    ----------
    x: numpy.array
        numpy array representing the one array to be multiplied
    y: numpy.array
        numpy array representing the second array to be multiplied
        if None it the module will square the x array

    Returns
    -------
    multi_array: numpy.array
                    numpy array representing the one to one multiplication
                    of two arrays
    '''
    if meta_data:
        m = meta_data[0]
        n = meta_data[1]
    else:
        m = 1
        n = 1
    if isinstance(y, str):
        multi_array = np.power(x, m)
        return multi_array
    else:
        multi_array = np.multiply(np.power(x, m), np.power(y, n))
        return multi_array


# list_1d1d = {
#         "1d_raw": transform_1d_none,
#         "1d_log": transform_1d_log,
#         "1d_exp": transform_1d_exp,
#         "1d_reciprocal": transform_1d_reciprocal,
#         "1d_cumsum": transform_1d_cumsum,
#         "1d_derivative": transform_1d_derivative,
#         "1d_multiply": transform_array_multiplication
#         }
# list_1d2d = {
#         "1d_cwt": transform_1d_cwt
#         }
