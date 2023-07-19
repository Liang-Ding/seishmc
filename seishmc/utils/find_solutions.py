# -------------------------------------------------------------------
# Functions to determine the solution
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

import pandas as pd
import numpy as np


def find_sol(df, type='misfit'):
    '''Determine the solution based on either:
    (1) The lowest Uq that is associated with waveform misfit (by default), or
    (2) Sample density.
    '''
    if type == 'misfit':
        return find_sol_misfit(df)
    else:
        return find_sol_sample_density(df)


def find_sol_sample_density(df):
    '''Find the solution based on sample density. '''

    keys = df.keys()
    sol = np.zeros(len(keys))
    bins = 25
    for i, k in enumerate(keys):
        hist, edges = np.histogram(df[k].values[:], bins=bins)
        sol[i] = edges[np.where(hist == np.max(hist))][0]
    return sol


def find_sol_misfit(df):
    '''Find the solution based on the lowest Uq (associated with waveform misfit) '''
    keys = df.keys()
    _sol = df[df['misfit'] == df['misfit'].min()]
    return [_sol[k].values[0] for k in keys]

