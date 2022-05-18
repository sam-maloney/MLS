#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:16:57 2022

@author: Samuel A. Maloney
"""

import numpy as np
from abc import ABCMeta, abstractmethod

class Basis(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): pass

    @abstractmethod
    def p(self, point):
        """Compute the basis polynomial values at a given set of points.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(n, ndim)
            Coordinates of evaluation points.

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(n, self.size)
            Values of basis polynomials at given points.

        """
        pass

    @abstractmethod
    def dp(self, point):
        """Compute the basis polynomial derivatives at a given point.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of evaluation points.

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(ndim, self.size)
            Derivatives of basis polynomials at the given point. The rows
            correspond to different spatial dimensions.

        """
        pass

    @abstractmethod
    def d2p(self, point):
        """Compute the basis polynomial 2nd derivatives at a given point.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of evaluation points.

        Returns
        -------
        numpy.ndarray, dtype='float64', shape=(ndim, self.size)
            2nd derivatives of basis polynomials at the given point. The rows
            correspond to different spatial dimensions.

        """
        pass

    def __call__(self, point):
        return self.p(point)

class LinearBasis(Basis):
    @property
    def name(self):
        return 'linear'

    def __init__(self, ndim):
        self.ndim = ndim
        self.size = ndim + 1
        self._dp = np.hstack((np.zeros((ndim,1)), np.eye(ndim)))
        self._d2p = np.zeros((ndim, ndim+1))

    def p(self, point):
        point.shape = (-1, self.ndim)
        return np.hstack((np.ones((len(point),1)), point))

    def dp(self, point=None):
        return self._dp

    def d2p(self, point=None):
        return self._d2p

class QuadraticBasis(Basis):
    @property
    def name(self):
        return 'quadratic'

    def __init__(self, ndim):
        self.ndim = ndim
        self.size = int((ndim+1)*(ndim+2)/2)
        self._dpLinear = np.hstack((np.zeros((ndim,1)), np.eye(ndim)))
        self._d2pLinear = np.zeros((ndim, ndim+1))

    def p(self, point):
        point.shape = (-1, self.ndim)
        if self.ndim == 1:
            return np.hstack((np.ones((len(point),1)), point, point**2))
        elif self.ndim == 2:
            x = point[:,0:1]
            y = point[:,1:2]
            return np.hstack((np.ones((len(point),1)), point, x**2, x*y, y**2))
        elif self.ndim == 3:
            x = point[:,0:1]
            y = point[:,1:2]
            z = point[:,2:3]
            return np.hstack((np.ones((len(point),1)), point,
                              x**2, x*y, x*z, y**2, y*z, z**2))

    def dp(self, point):
        point.shape = (self.ndim,)
        if self.ndim == 1:
            return np.hstack((self._dpLinear,[2.*point]))
        elif self.ndim == 2:
            x = point[0]
            y = point[1]
            return np.hstack((self._dpLinear,[[2.*x, y,  0. ],
                                              [ 0. , x, 2.*y]]))
        elif self.ndim == 3:
            x = point[0]
            y = point[1]
            z = point[2]
            return np.hstack((self._dpLinear,[[2.*x, y , z ,  0. , 0.,  0. ],
                                              [ 0. , x , 0., 2.*y, z ,  0. ],
                                              [ 0. , 0., x ,  0. , y , 2.*z]]))

    def d2p(self, point):
        if self.ndim == 1:
            return np.hstack((self._d2pLinear, [[2.]]))
        elif self.ndim == 2:
            return np.hstack((self._d2pLinear, [[2., 0., 0.],
                                                [0., 0., 2.]]))
        elif self.ndim == 3:
            return np.hstack((self._d2pLinear,[[2., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 2., 0., 0.],
                                               [0., 0., 0., 0., 0., 2.]]))
