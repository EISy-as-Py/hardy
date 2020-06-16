"""
TEST Transformations.py
COPIED DOCSTRING FROM MAIN Transformations.py planning header

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
# import os
# import sys
# import csv
# import time
# import math
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
        # assert isinstance(type(result), np.ndarray), "The output result\
        #        type is not correct"

    def test_transform_1d_reciprocal(self):
        test_array = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = tforms.transform_1d_reciprocal(test_array)

        assert len(test_array) == len(result), "The returned size for\
                array is invalid"
        # assert isinstance(type(result), np.ndarray), "The output result\
        #        type is not correct"
        # assert isinstance(type(result[0]), np.float64), "The output\
        #        result type is not correct"

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
