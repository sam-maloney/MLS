# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:55:08 2020

@author: samal
"""

import numpy as np
import matplotlib.pyplot as plt

from WeightFunction import *

# x = np.linspace(0,1,101)
x = np.linspace(-1,1,201)

##### Values #####
plt.plot(x, Bump()(np.abs(x)), label='bump')
plt.plot(x, Gaussian()(np.abs(x)), label='Gaussian')
plt.plot(x, QuinticSpline()(np.abs(x)), label='quintic')
plt.plot(x, SimpleQuinticSpline()(np.abs(x)), label='simpleQuintic')
plt.plot(x, QuarticSpline()(np.abs(x)), label='quartic')
plt.plot(x, CubicSpline()(np.abs(x)), label='cubic')
plt.plot(x, QuadraticSpline()(np.abs(x)), label='quadratic')
plt.plot(x, SimpleCubicSpline()(np.abs(x)), label='simpleCubic')


##### 1st Derivatives #####
# plt.plot(x, Bump().dw(np.abs(x))[1], label='bump')
# plt.plot(x, Gaussian().dw(np.abs(x))[1], label='Gaussian')
# plt.plot(x, QuinticSpline().dw(np.abs(x))[1], label='quintic')
# plt.plot(x, SimpleQuinticSpline().dw(np.abs(x))[1], label='simpleQuintic')
# plt.plot(x, QuarticSpline().dw(np.abs(x))[1], label='quartic')
# plt.plot(x, CubicSpline().dw(np.abs(x))[1], label='cubic')
# plt.plot(x, QuadraticSpline().dw(np.abs(x))[1], label='quadratic')
# plt.plot(x, SimpleCubicSpline().dw(np.abs(x))[1], label='simpleCubic')

##### 2nd Derivatives #####
# plt.plot(x, Bump().d2w(np.abs(x))[2], label='bump')
# plt.plot(x, Gaussian().d2w(np.abs(x))[2], label='Gaussian')
# plt.plot(x, QuinticSpline().d2w(np.abs(x))[2], label='quintic')
# plt.plot(x, SimpleQuinticSpline().d2w(np.abs(x))[2], label='simpleQuintic')
# plt.plot(x, QuarticSpline().d2w(np.abs(x))[2], label='quartic')
# plt.plot(x, CubicSpline().d2w(np.abs(x))[2], label='cubic')
# plt.plot(x, QuadraticSpline().d2w(np.abs(x))[2], label='quadratic')
# plt.plot(x, SimpleCubicSpline().d2w(np.abs(x))[2], label='simpleCubic')

plt.legend()

# plt.savefig(f"weight_functions_plot.svg", bbox_inches = 'tight', pad_inches = 0)