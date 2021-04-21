import copy
import datetime
import sys
# sys.path.append(
# r'C:\Users\lacho\Anaconda3\envs\sasmodels_test\Lib\site-packages\sasview\src')
sys.path.append(r'./sasview/src')

# sys.path.append(r'$SOURCE/Desktop/DIRECT/sasmodels/sasview/src')
import matplotlib.pyplot as plt             # noqa: F402
import numpy as np                          # noqa: F402
import pandas as pd                         # noqa: F402
from bumps.names import *                   # noqa: F402
from bumps.fitters import fit               # noqa: F402
import pickle                               # noqa: F402
import sasmodels                            # noqa: F402
# from ouzosamplehandling import *

from sasmodels.core import load_model       # noqa: F402
from sasmodels.bumps_model import Model, Experiment     # noqa: F402
from sasmodels.data import load_data, plot_data, empty_data1D      # noqa: F402
from sasmodels.direct_model import DirectModel          # noqa: F402
from sas.sascalc.fit.qsmearing import smear_selection   # noqa: F402

import sas                                  # noqa: F402
import os                                   # noqa: F402
# %matplotlib inline
import random                               # noqa: F402


def scat_model(data, label):

    if label == "ellipsoid":
        pars = dict(
            scale=1.0, background=0.001,
            )
        kernel = load_model(label)
        model = Model(kernel, **pars)

        # SET THE FITTING PARAMETERS
        model.radius_polar.range(0.0, 1000.0)
        model.radius_equatorial.range(0.0, 1000.0)
        model.sld.range(-0.56, 8.00)
        model.sld_solvent.range(-0.56, 6.38)
        model.radius_polar_pd.range(0, 0.11)
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)
        result = fit(problem, method='dream')
        chisq = problem.chisq()

    if label == "shell":
        label = "core_shell_sphere"
        pars = dict(
            scale=1.0, background=0.001,
            )
        kernel = load_model(label)
        model = Model(kernel, **pars)

        # SET THE FITTING PARAMETERS
        model.radius.range(0.0, 1000.0)
        model.thickness.range(0.0, 100.0)
        model.sld_core.range(-0.56, 8.00)
        model.sld_shell.range(-0.56, 8.00)
        model.sld_solvent.range(-0.56, 6.38)
        model.radius_pd.range(0.1, 0.11)
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)
        result = fit(problem, method='dream')
        chisq = problem.chisq()

    if label == "cylinder":
        pars = dict(
            scale=1.0, background=0.001,
            )
        kernel = load_model(label)
        model = Model(kernel, **pars)

        # SET THE FITTING PARAMETERS
        model.radius.range(0, 1000.0)
        model.length.range(0, 1000.0)
        model.sld.range(-0.56, 8.00)
        model.sld_solvent.range(-0.56, 6.38)
        model.radius_pd.range(0, 0.11)
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)
        result = fit(problem, method='dream')
        chisq = problem.chisq()

    if label == "sphere":
        pars = dict(
            scale=1.0, background=0.001,
            )
        kernel = load_model(label)
        model = Model(kernel, **pars)

        # SET THE FITTING PARAMETERS
        model.radius.range(0.0, 3200.0)
        model.sld.range(-0.56, 8.00)
        model.sld_solvent.range(-0.56, 6.38)
        model.radius_pd.range(0.1, 0.11)
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)
        result = fit(problem, method='dream')
        chisq = problem.chisq()

    return np.round(chisq, 3)


def model_evaluation(dataframe, classes, datapath='./', filename='results'):

    pred_chisq = []
    alternative_chisq = []

    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        filepath = datapath+row[0]+'.csv'

        data = load_data(filepath)

        probab = [item for item in row[3].split('[')[1].split(
            ']')[0].split(' ') if item != '']

        probab = list(map(np.float64, probab))
        max_label = [probab.index(x) for x in sorted(probab, reverse=True)[:2]]
        label = classes[max_label[0]]

        chi_squared = scat_model(data, label)
        pred_chisq.append((label, chi_squared))

        if probab[max_label[0]] >= 0.7:
            alternative_chisq.append('-')
        else:
            label = classes[max_label[1]]
            chi_squared = scat_model(data, label)
            alternative_chisq.append((label, chi_squared))

    dataframe['predicted_chi_square'] = pred_chisq
    dataframe['alternative_chi_square'] = alternative_chisq

    dataframe.to_csv(filename, index=False)
    return dataframe
