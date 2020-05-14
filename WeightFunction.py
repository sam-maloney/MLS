# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:55:16 2020

@author: samal

Classes
-------
WeightFunction : metaclass=ABCMeta
    Abstract base class for weight function implementations. All functions are
    defined to be zero-valued for :math:`r \geq 1`.
QuadraticSpline : WeightFunction
    .. math::
        w(r)=
        \\begin{cases}
        -2r^2 + 1, & r < 0.5 \\\\
        2r^2 - 4r + 2, & 0.5 \\leq r \\leq 1
        \\end{cases}
SimpleCubicSpline : WeightFunction
    .. math:: w(r) = 2r^3 - 3r^2 + 1, \\quad r < 1
CubicSpline : WeightFunction
    .. math::
        w(r)=
        \\begin{cases}
        6r^3 - 6r^2 + 1, & r < 0.5 \\\\
        -2r^3 + 6r^2 - 6r + 2, & 0.5 \\leq r < 1
        \\end{cases}
QuarticSpline : WeightFunction
    .. math:: w(r) = -3r^4 + 8r^3 - 6r^2 + 1
QuinticSpline : WeightFunction
    .. math:: w(r) = -6r^5 + 15r^4 - 10r^3 + 1
SimpleQuinticSpline : WeightFunction
    .. math:: w(r) = -6r^5 + 15r^4 - 10r^3 + 1
Gaussian : WeightFunction
    .. math:: w(r) = \\frac{\\exp(-9r^2) - \\exp(-9)}{1.0 - \\exp(-9)}
"""

from abc import ABCMeta, abstractmethod
import numpy as np

class WeightFunction(metaclass=ABCMeta):
    @property
    @abstractmethod
    def form(self): pass

    @abstractmethod
    def w(self, r):
        """Compute kernel weight function value.
    
        Parameters
        ----------
        r : numpy.ndarray, dtype='float64', shape=(n,)
            Distances from evaluation points to node point.
    
        Returns
        -------
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the kernel function at the given distances.
    
        """
        pass
    
    @abstractmethod
    def dw(self, r):
        """Compute kernel weight function value and its radial derivative.

        Parameters
        ----------
        r : numpy.ndarray, dtype='float64', shape=(n,)
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the kernel function at the given distances.
        dwdr : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the radial derivative at the given distances.

        """
        pass
    
    @abstractmethod
    def d2w(self, r):
        """Compute kernel weight function and its radial derivatives.

        Parameters
        ----------
        r : numpy.ndarray, dtype='float64', shape=(n,)
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the kernel function at the given distances.
        dwdr : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.ndarray, dtype='float64', shape=(n,)
            Values of the 2nd order radial derivative at the given distances.

        """
        pass
    
    def __call__(self, r):
        return self.w(r)

class QuadraticSpline(WeightFunction):
    @property
    def form(self):
        return 'quadratic'
    
    def w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r <= 1, i0)
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]
            w[i0] = -2.*r1*r1 + 1.
        if i1.any():
            r1 = r[i1]
            w[i1] = 2.*r1*r1 - 4.*r1 + 2.
        return w
    
    def dw(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r <= 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            w[i0] = -2.*r1*r1 + 1.
            dwdr[i0] = -4.*r1
        if i1.any():
            r1 = r[i1]
            w[i1] = 2.*r1*r1 - 4.*r1 + 2.
            dwdr[i1] = 4.*r1 - 4.
        return w, dwdr
    
    def d2w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r <= 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            w[i0] = -2.*r1*r1 + 1.
            dwdr[i0] = -4.*r1
            d2wdr2[i0] = -4.
        if i1.any():
            r1 = r[i1]
            w[i1] = 2.*r1*r1 - 4.*r1 + 2.
            dwdr[i1] = 4.*r1 - 4.
            d2wdr2[i1] = 4.
        return w, dwdr, d2wdr2

class SimpleCubicSpline(WeightFunction):
    @property
    def form(self):
        return 'simpleCubic'
    
    def w(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 2*r3 - 3*r2 + 1
        return w
    
    def dw(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 2*r3 - 3*r2 + 1
            dwdr[i0] = 6*r2 - 6*r1
        return w, dwdr
    
    def d2w(self, r):
        i0 = r <= 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 2*r3 - 3*r2 + 1
            dwdr[i0] = 6*r2 - 6*r1
            d2wdr2[i0] = 12*r1 - 6
        return w, dwdr, d2wdr2

class CubicSpline(WeightFunction):
    @property
    def form(self):
        return 'cubic'
    
    def w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 6*r3 - 6*r2 + 1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1
            w[i1] = -2*r3 + 6*r2 - 6*r1 + 2
        return w
    
    def dw(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 6*r3 - 6*r2 + 1
            dwdr[i0] = 18*r2 - 12*r1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1
            w[i1] = -2*r3 + 6*r2 - 6*r1 + 2
            dwdr[i1] = -6*r2 + 12*r1 - 6
        return w, dwdr
    
    def d2w(self, r):
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1
            w[i0] = 6*r3 - 6*r2 + 1
            dwdr[i0] = 18*r2 - 12*r1
            d2wdr2[i0] = 36*r1 - 12
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1
            w[i1] = -2*r3 + 6*r2 - 6*r1 + 2
            dwdr[i1] = -6*r2 + 12*r1 - 6
            d2wdr2[i1] = -12*r1 + 12
        return w, dwdr, d2wdr2

class QuarticSpline(WeightFunction):
    @property
    def form(self):
        return 'quartic'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2
            w[i0] = -3*r4 + 8*r3 - 6*r2 + 1
        return w
    
    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2
            w[i0] = -3*r4 + 8*r3 - 6*r2 + 1
            dwdr[i0] =  -12*r3 + 24*r2 - 12*r1
        return w, dwdr
    
    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2
            w[i0] = -3*r4 + 8*r3 - 6*r2 + 1
            dwdr[i0] =  -12*r3 + 24*r2 - 12*r1
            d2wdr2[i0] = -36*r2 + 48*r1 - 12
        return w, dwdr, d2wdr2

class QuinticSpline(WeightFunction):
    @property
    def form(self):
        return 'quintic'

    def w(self, r):
        i0 = r < 1/3
        i1 = np.logical_xor(r < 2/3, i0)
        i2 = np.logical_xor(r < 1, i0 + i1)
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -405/11*r5 + 405/11*r4 - 90/11*r2 + 1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i1] = 405/22*r5 - 1215/22*r4 + 675/11*r3 - 315/11*r2 + 75/22*r1 + 17/22
        if i2.any():
            r1 = r[i2]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i2] = -81/22*r5 + 405/22*r4 - 405/11*r3 + 405/11*r2 - 405/22*r1 + 81/22
        return w
    
    def dw(self, r):
        i0 = r < 1/3
        i1 = np.logical_xor(r < 2/3, i0)
        i2 = np.logical_xor(r < 1, i0 + i1)
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -405/11*r5 + 405/11*r4 - 90/11*r2 + 1
            dwdr[i0] = -2025/11*r4 + 1620/11*r3 - 180/11*r1
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i1] = 405/22*r5 - 1215/22*r4 + 675/11*r3 - 315/11*r2 + 75/22*r1 + 17/22
            dwdr[i1] = 2025/22*r4 - 2430/11*r3 + 2025/11*r2 - 630/11*r1 + 75/22
        if i2.any():
            r1 = r[i2]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i2] = -81/22*r5 + 405/22*r4 - 405/11*r3 + 405/11*r2 - 405/22*r1 + 81/22
            dwdr[i2] = -405/22*r4 + 810/11*r3 - 1215/11*r2 + 810/11*r1 - 405/22
        return w, dwdr
    
    def d2w(self, r):
        i0 = r < 1/3
        i1 = np.logical_xor(r < 2/3, i0)
        i2 = np.logical_xor(r < 1, i0 + i1)
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -405/11*r5 + 405/11*r4 - 90/11*r2 + 1
            dwdr[i0] = -2025/11*r4 + 1620/11*r3 - 180/11*r1
            d2wdr2[i0] = -8100/11*r3 + 4860/11*r2 - 180/11
        if i1.any():
            r1 = r[i1]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i1] = 405/22*r5 - 1215/22*r4 + 675/11*r3 - 315/11*r2 + 75/22*r1 + 17/22
            dwdr[i1] = 2025/22*r4 - 2430/11*r3 + 2025/11*r2 - 630/11*r1 + 75/22
            d2wdr2[i1] = 4050/11*r3 - 7290/11*r2 + 4050/11*r1 - 630/11
        if i2.any():
            r1 = r[i2]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i2] = -81/22*r5 + 405/22*r4 - 405/11*r3 + 405/11*r2 - 405/22*r1 + 81/22
            dwdr[i2] = -405/22*r4 + 810/11*r3 - 1215/11*r2 + 810/11*r1 - 405/22
            d2wdr2[i2] = -810/11*r3 + 2430/11*r2 - 2430/11*r1 + 810/11
        return w, dwdr, d2wdr2

class SimpleQuinticSpline(WeightFunction):
    @property
    def form(self):
        return 'simpleQuintic'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -8/3*r5 + 5*r4 - 10/3*r2 + 1
        return w
    
    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -8/3*r5 + 5*r4 - 10/3*r2 + 1
            dwdr[i0] = -40/3*r4 + 20*r3 - 20/3*r1
        return w, dwdr
    
    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1*r1; r3 = r2*r1; r4 = r2*r2; r5 = r3*r2
            w[i0] = -8/3*r5 + 5*r4 - 10/3*r2 + 1
            dwdr[i0] = -40/3*r4 + 20*r3 - 20/3*r1
            d2wdr2[i0] = -160/3*r3 + 60*r2 - 20/3
        return w, dwdr, d2wdr2

class Gaussian(WeightFunction):
    c1 = np.exp(-9.0)
    c2 = 1.0/(1.0 - np.exp(-9.0))
    
    @property
    def form(self):
        return 'gaussian'

    def w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        if i0.any():
            r2 = r[i0]**2
            w[i0] = (np.exp(-9.0*r2) - self.c1) / (1.0 - self.c1)
        return w
    
    def dw(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1**2
            w[i0] = (np.exp(-9.0*r2) - self.c1) * self.c2
            dwdr[i0] = -18.0*r1*np.exp(-9.0*r2) * self.c2
        return w, dwdr
    
    def d2w(self, r):
        i0 = r < 1
        w = np.zeros(r.size)
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]; r2 = r1**2
            w[i0] = (np.exp(-9.0*r2) - self.c1) * self.c2
            dwdr[i0] = -18.0*r1*np.exp(-9.0*r2) * self.c2
            d2wdr2[i0] = 18.0*np.exp(-9.0*r2)*(18.0*r2-1) * self.c2
        return w, dwdr, d2wdr2