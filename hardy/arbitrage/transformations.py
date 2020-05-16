# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:04:12 2020
@author: hurtd

The first-level functions, which will take input data
    (either 2D, 3D, or eventually nD...), and perform transformations
    to generate the full set of data-columns that we will test against!

THIS FILE IS FOR MATHEMATICAL TRANSFORMATIONS THEMSELVES.
DATA WILL BE PROCESSED IN THE "ARBITRAGE.py" FILE

 This package will contain several 'sections':
  * __Transformation Functions:__ This is the mathematical side,
              and to start out we will be able to perform a variety
              of 1D or 2D transformations such as Log, Inverse, accumulate,
              Integrate, derrive, etc.
  * __Complex Transforms__: Some data transformations are combinations of
              the ones above (you can integrate AFTER you log-ify,
              for instance)


 __Timeline + Milestones__:
  * 2020-04-21: List of the high-priority functions and
                  Simple-Transformations, with progress and
                  timeline to get them all done soon.
  * 2020-04-28: Passing Tests and can __HAND OFF__ to the classifier - a
                  DataFrame of "all" the transformed data columns.
                  Recieve Handoff from handling, and begin to Integrate.
  * 2020-05-12: Complex Transforms - consider what other things we may want,
                  and discuss feedback with group
  * 2020-06-09: Make Decision on Association functions and __HAND OFF__
                  if so. Otherwise, simply focus on new group priorities
  * 2020-06-23: IF yNot function is doing Configuration ideas, make
                  Stretch-Goal learning gameplan... TBD...

 __Current Status__:
  * (2020-04-14)
  * Just creating files and setup, no progress yet
  * Planning Functions, in compontent spec document (here!)
  * Considering how much to Frankenstien from prior work. Configuration?

 __Module List__:

 ### *SECTION: Basic 1D Transformations*
#### transform_log():
  * INPUT: 1D data array with NO NEGATIVE VALUES
  * OUTPUT: Logrythmic transform of that data
  * Note: Consider shifting or abs() for negative data?
          no? simply don't call log transform for negative data?

#### transform_reciprocal():
  * INPUT: 1D data array - Limits tbd?
  * OUTPUT: all values inverted (1/x)
  * Note: twice should return itself!

#### transform_cumsum():
  * INPUT: 1D data array -
  * OUTPUT: cumulative sum of data (aka integrated with unit-steps)
  * Note: Not in high priority list?

#### transform_1d_derrivative():
  * INPUT: 1D data array
  * OUTPUT: the step-by-step delta (Note: copy last delta to retain length?)
  * Note: Also not in high-priority list? Should also be able to complete
          the loop w/ cumsum.
#### transform_exp():
  * INPUT: 1D data array - Limits?
  * OUTPUT: e^x of each datapoint.
  * Note: may be redundant in general with log? should be able to
          complete that loop!

#### transform_0to1():
  * INPUT: 1D data array, data handling case instructions
  * OUTPUT: that array shifted and scaled to the 0-to-1 basis
          (by FIRST shifting to min=0, THEN scaling to max=1)
  * Note: option to leave data alone if min is already 0-to-1,
          or if max After Shift is already 0-to-1
          (case: data begins 0.2-0.4, can either scale 0-1 or leave as is!)
  * Note2: BETTER scaling is to devine 0 as -MAX(ABS(Data))
          and 1 as +MAX(ABS(Data)). This maintains fidelity of absolute Y data

#### transform_pm1():
  * INPUT: 1D data array, data handling case instructions
  * OUTPUT: that array shifted and scaled to the 0-to-1 basis
          [THIS TIME: 0=(-,MaxAbs(X)), and 1 =(+MaxAbs(X))]


### *SECTION: Basic 2D Transformations*

#### transform_2D_int():
  * INPUT: 2 equal size 1D arrays Y, X, to be integrated (Y)dx -
          [Optional offset value? to use as the Plus-C]
  * OUTPUT: The integral of Y dx (BOX? Trapz?) - offset if instructed to.
  * Note: Error handling? What if not Sorted/Linear in X?
          Should we sort by X first? (Or, what if reciprocating
          data ie CV Sweeps?)

#### transform_2D_der():
  * INPUT: 2 equal size 1D arrays Y, X - Sorted? in either?
  * OUTPUT: the Single-point derivatives dY/dX, also the offset so
          you /could/ integrate it back again!
  * Note: could use some sort of average or smoothing to reduce noise?
          However that would be LOSSY DATA PRACTICE!

#### transform_prod():
  * INPUT: 2 equal size 1D arrays X, Y , [Optional Power arguments?
          or do those in the 1D cases and use as inputs?]
  * OUTPUT: Product of each x*y, maybe with power-math included (options)

### *SECTION: More Complex Transformations*
 #### transform_fourier_wavelets():
  * Ok so this is the only High-Priority one that I'm genuinely
          concerned with... while you "CAN" try to do a transform on
          a whole dataset, that gets noisy and lossy.
    What I want to investigate is "Wavelet Filtering" Fourier transform,
          which we learned about at a Data Sci seminar last quarter?
          (Or otherwise, there's a whole realm of Signal-transforming
          science, I can research that...)
  * INPUT:  2 equal size 1D arrays X, Y - Sorted in X??
          (Frequency range parameters? or is that the X-size?)
  * OUTPUT: 2D? output matrix or Meshgrid - in X-Freq space
          (for each wavelet size, return the match(-1 to 1?)*amplitude
          at each X?)
  * NOTE: This will have to be a group discussion- we need TEST DATA
          that should work in this space, and then we can report that back!

 #### multi_transform():
  * Wrapping function, to perform multiple transformations all together...
          Not sure which of these may be useful but I can see possible
          value in knowing the integral of a log function, for example.
  * INPUT: X, [Y if 2D], Multiple transforms to perform...
          Is this what classes are for??
  * OUTPUT: Data output from the final transform listed.
  * NOTE: This is low-priority, and should only be done if we convince
          ourselves that it's useful... RELATED, if we get the "Smart"
          learning functionality, maybe we can combine things this way

 #### transform___():
  *
  * INPUT:
  * OUTPUT:

"""
# asdf

# asdf

# asdf

# asdf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def transform_1d_exp(raw_array):

    return np.exp(raw_array)


def transform_1d_log(raw_array):
    '''The function that outputs the natural log of input array

    PARAMETERS:
    -----------
    raw_array: Input numpy array

    RETURNS:
    --------
    log_array: np.ndarray
               natural log values of each element in the input array
    '''
    log_array = np.log(raw_array)
    return log_array


def transform_1d_reciprocal(raw_array):
    '''The function the outputs the reciprocal of input array

    PARAMETERS:
    -----------
    raw_array: Input numpy array

    RETURNS:
    --------
    log_array: np.ndarray
               reciprocal values of each element in the input array
    '''
    reciprocal_array = np.reciprocal(raw_array)
    return reciprocal_array


def transform_1d_cumsum(raw_array):
    '''The function return the cumulative sum of input array

    PARAMETERS:
    -----------
    raw_array: Input numpy array

    RETURNS:
    --------
    log_array: np.ndarray
               cumulative sum of values in the input array
    '''
    cumsum_array = np.cumsum(raw_array)
    return cumsum_array


def transform_1d_derivative(raw_array, spacing=0):
    ''' Function that outputs the gradient of 1-D array using
    numpy.gradient function

    PARAMETERS:
    -----------
    raw_array: numpy array
    spacing: int representing the spacing between each datapoint

    RETURNS:
    --------
    derivative_array: np.ndarray
                      array representing gradient at each datapoint
    '''

    if spacing == 0:
        spacing = np.arange(np.size(raw_array))
    else:
        spacing = spacing

    derivative_array = np.gradient(raw_array, spacing)

    return derivative_array


def transform_1d_cwt(raw_df, xy=0):
    """
    Transform to execute a "Continuous Wavelet Transform" on a 1d data array
    pass it a raw XY data and tell it which column to use for the transform.
    See Documentaion on CWT transform:
            https://docs.scipy.org/doc/scipy/reference/generated/
            scipy.signal.cwt.html#scipy.signal.cwt
    Note: I need to do testing to understand the in/outputs here...
        *Plan is to simply hard-code a certain type of Wavelet to use... and
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
    # cwt_matrix = signal.cwt(data, signal.ricker, widths)
    # Optional different Signal to compare with: "Morlet2" but not working?)
    cwt_matrix = signal.cwt(data, signal.morlet, widths)

    return cwt_matrix


list_1d = {
        "1d_log": transform_1d_log,
        "1d_reciprocal": transform_1d_reciprocal,
        "1d_cumsum": transform_1d_cumsum,
        "1d_derivative": transform_1d_derivative,
        "1d_cwt": transform_1d_cwt
        }


#=============================================================================
# test
# x_linear = np.linspace(0, 20, 1000)
# y_test = 10 * np.sin(2 * np.pi * 0.1 * x_linear) + 1 * np.sin(2 * np.pi * 5 * x_linear)
# test_df = pd.DataFrame(data={"Xlinear": x_linear, "Ytest": y_test})
# Valid Test: Pass Test_df and the Y-axis transform to get output df
# of the Y-test data.
# fig, ax = plt.subplots(2, 1)
# result_1 = transform_1d_cwt(test_df, 'y')
# ax[0].imshow(result_1, cmap='PRGn')
# ax[1].plot(x_linear, y_test)
# fig, ax2 = plt.subplots(2, 1)
# y_chirp = signal.chirp(x_linear, 2, 20, 0.001)
# result_2 = transform_1d_cwt(y_chirp)
# ax2[0].imshow(result_2, cmap='PRGn')
# ax2[1].plot(x_linear, y_chirp)
    #=============================================================================
