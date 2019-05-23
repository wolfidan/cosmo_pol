#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:23:27 2019

@author: wolfensb
"""

from scipy.interpolate import interp2d
import pandas as pd
import numpy as np


def nyquist_interpolator(nyquist_file):
    data_nyquist = pd.read_csv(nyquist_file)

    def interpolator(elevation, azimuth):
        closest_el = data_nyquist['elevation'][np.argmin(
                np.abs(data_nyquist['elevation'] - elevation))]

        idx_el = np.where(data_nyquist['elevation'] == closest_el)[0]
        az_at_el = data_nyquist['azimuth'][idx_el]

        closest_az = data_nyquist['azimuth'][np.argmin(
                data_nyquist['azimuth'][az_at_el] - azimuth)]

        nyq = data_nyquist['nyquist'][idx_el[np.where(closest_az ==
                          data_nyquist['azimuth'][az_at_el])[0][0]]]
        return nyq
    return interpolator
