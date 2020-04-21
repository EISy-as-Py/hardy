"""
Y Not?
A module contianing useful functions that don't otherwise fit in our package
DH 2020-04-20

Contents:
    1) Arbitrary Data set generation based on mathematical operations
        * Linear data, Sinusoidal data, exponential data, etc.
        * Generate an XY set of specified length, and save it as a file
        * Do this in a specific folder structure using ../local_data/
        * Loop for thousands (as needed) of files
"""
import pandas as pd
import numpy as np

default_location = "../local_data/"


def generate_linear(length=64, xrange=[0,1], m=1.0, b=1.0):
    """
    Given set parmeters, generate an XY dataframe of x and y=mx+b

    Parameters
    ----------
    length : int
            The length of the data file to produce
    xrange : list or tuple, length 2
            X minimum and maximum values - range to be used for x data
    m : float64
            slope of the line to be created
    b : float64
            intercept of line to be created.

    Returns
    -------
    result : pandas.dataframe
            simple x,y dataframe to be written
    """
    x_array = np.linspace(min(xrange),max(xrange),int(length))
    y_array = x_array * m + b
    result = pd.DataFrame()
    result['X']=x_array
    result['Y=mx+b'] = y_array
    return result


def generate_sin(length=64, xrange=[0,1], A=1.0, f=1.0, theta=0.0):
    """
    Given set parmeters, generate an XY dataframe of x and
    y=A*sin(2pi*f*(x-theta))

    Parameters
    ----------
    length : int
            The length of the data file to produce
    xrange : list or tuple, length 2
            X minimum and maximum values - range to be used for x data
    A : float64
            Amplitude of sin wave
    f : float64
            frequency of the set, scaled by 2Pi so that at f=1, there will be
            one full period of the wave from 0 to 1
    theta : float64
            phase offset, scaled by 2Pi so that at theta=1, there will be one
            full period offset meaning no actual offset at all... At 0.25,
            the sine wave would become a cosine wave! (or at -0.25...)

    Returns
    -------
    result : pandas.dataframe
            simple x,y dataframe to be written
    """
    x_array = np.linspace(min(xrange),max(xrange),int(length))
    y_array = A * np.sin(2 * np.pi * f * (x_array - theta))
    result = pd.DataFrame()
    result['X']=x_array
    result['y=sin_x'] = y_array
    return result


def generate_log(length=64, xrange=[0,1], A=1.0, y0=0, x0=-0.5):
    """
    Given set parmeters, generate an XY dataframe of x and
    y=A*sin(2pi*f*(x-theta))

    Parameters
    ----------
    length : int
            The length of the data file to produce
    xrange : list or tuple, length 2
            X minimum and maximum values - range to be used for x data
    A : float64
            Y-stretching multiplier
    y0 : float64
            y-translation
    x0 : float64
            x-translation. MUST be MORE negative than Xmin because log(-1) is
            nothing.

    Returns
    -------
    result : pandas.dataframe
            simple x,y dataframe to be written
    """
    x_array = np.linspace(min(xrange), max(xrange), int(length))
    if min(xrange) < x0:
        x0 = min(xrange) - 0.1
    y_array = A * np.log(x_array - x0) + y0
    result = pd.DataFrame()
    result['X']=x_array
    result['y=log_x'] = y_array
    return result





