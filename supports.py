#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:08:13 2022

@author: Samuel A. Maloney
"""

import numpy as np
from scipy.linalg import norm
from abc import ABCMeta, abstractmethod

class Support(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): pass

    def __init__(self, mlsSim, size):
        self.sim = mlsSim
        self.ndim = mlsSim.ndim
        self.weightFunction = mlsSim.weightFunction
        self.size = size
        self.rsize = 1./size

    @abstractmethod
    def w(self, point):
        """Compute kernel function values and support indices at given point.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of kernel for all n nodes in self.sim.nodes[indices].

        """
        pass

    @abstractmethod
    def dw(self, point):
        """Compute kernel values, gradients, and support indices at point.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of kernel for all n nodes in self.sim.nodes[indices].
        gradw : numpy.ndarray, dtype='float64', shape=(n,ndim)
            Gradients of kernel for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dx1,dy1,dz1],[dx2,dy2,dz2]...])

        """
        pass

    @abstractmethod
    def d2w(self, point):
        """Compute kernel values, gradients, laplacians, and indices at point.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        w : numpy.ndarray, dtype='float64', shape=(n,)
            Values of kernel for all n nodes in self.sim.nodes[indices].
        gradw : numpy.ndarray, dtype='float64', shape=(n,ndim)
            Gradients of kernel for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dx1,dy1,dz1],[dx2,dy2,dz2]...])
        grad2w :  numpy.ndarray, dtype='float64', shape=(n,ndim)
            2nd derivatives of kernel for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dxx1,dyy1,dzz1],[dxx2,dyy2,dzz2]...]).

        """
        pass

    def __call__(self, point):
        return self.w(point)

    def __repr__(self):
        return f"('{self.name}', {self.size*self.sim.N})"

class CircularSupport(Support):
    @property
    def name(self):
        return 'circular'

    def __init__(self, mlsSim, size):
        super().__init__(mlsSim, size)
        factor = [2., np.pi, 4.*np.pi/3.][self.ndim-1]
        self.volume = factor*(self.size + 0.5/mlsSim.N)**self.ndim

    def w(self, point):
        distances = norm(point - self.sim.nodes, axis=1)
        indices = np.flatnonzero(distances < self.size).astype('uint32')
        w = self.weightFunction.w(distances[indices] * self.rsize)
        return indices, w

    def dw(self, point):
        indices = np.flatnonzero(norm(point - self.sim.nodes, axis=1)
                                 < self.size).astype('uint32')
        displacements = (point - self.sim.nodes[indices]) * self.rsize
        distances = np.array(norm(displacements, axis=-1))
        w, dwdr = self.weightFunction.dw(distances)
        gradr = np.full(displacements.shape, np.sqrt(1.0/self.ndim)*self.rsize,
                        dtype='float64')
        i = distances > 1e-14
        gradr[i] = displacements[i] / (distances[i]*self.size).reshape((-1,1))
        gradw = dwdr.reshape((-1,1)) * gradr
        return indices, w, gradw

    def d2w(self, point):
        indices = np.flatnonzero(norm(point - self.sim.nodes, axis=1)
                                 < self.size).astype('uint32')
        displacements = (point - self.sim.nodes[indices]) * self.rsize
        distances = np.array(norm(displacements, axis=-1))
        w, dwdr, d2wdr2 = self.weightFunction.d2w(distances)
        gradr = np.full(displacements.shape, np.sqrt(1.0/self.ndim)*self.rsize,
                        dtype='float64')
        i = distances > 1e-14
        gradr[i] = displacements[i] / (distances[i]*self.size).reshape((-1,1))
        gradw = dwdr.reshape((-1,1)) * gradr
        grad2w = d2wdr2.reshape((-1,1)) * gradr*gradr
        return indices, w, gradw, grad2w

class RectangularSupport(Support):
    @property
    def name(self):
        return 'rectangular'

    def __init__(self, mlsSim, size):
        super().__init__(mlsSim, size)
        self.volume = (2.*(self.size + 0.5/mlsSim.N))**self.ndim

    def w(self, point):
        distances = np.abs(point - self.sim.nodes)
        indices = np.flatnonzero((distances < self.size).all(axis=1)) \
                                .astype('uint32')
        w = np.prod( np.apply_along_axis( self.weightFunction.w, 0,
                distances[indices] * self.rsize ), axis=1 )
        return indices, w

    def dw(self, point):
        displacements = point - self.sim.nodes
        distances = np.abs(displacements)
        indices = np.flatnonzero((distances < self.size).all(axis=1)) \
                                .astype('uint32')
        w = np.ones((len(indices), self.ndim))
        dwdr = np.empty(w.shape)
        for i in range(self.ndim):
            w[:,i], dwdr[:,i] = self.weightFunction.dw(distances[indices,i]
                                                       * self.rsize)
        gradw = dwdr * np.sign(displacements[indices]) * self.rsize
        if self.ndim == 2:
            gradw[:,0] *= w[:,1]
            gradw[:,1] *= w[:,0]
        elif self.ndim == 3:
            gradw[:,0] *= w[:,1]*w[:,2]
            gradw[:,1] *= w[:,0]*w[:,2]
            gradw[:,2] *= w[:,0]*w[:,1]
        w = np.prod(w, axis=1)
        return indices, w, gradw

    def d2w(self, point):
        displacements = point - self.sim.nodes
        distances = np.abs(displacements)
        indices = np.flatnonzero((distances < self.size).all(axis=1)) \
                                .astype('uint32')
        w = np.ones((len(indices), self.ndim))
        dwdr = np.empty(w.shape)
        d2wdr2 = np.empty(w.shape)
        for i in range(self.ndim):
            w[:,i], dwdr[:,i], d2wdr2[:,i] = \
                self.weightFunction.d2w(distances[indices,i] * self.rsize)
        gradw = dwdr * np.sign(displacements[indices]) * self.rsize
        grad2w = d2wdr2 * self.rsize**2
        if self.ndim == 2:
            gradw[:,0] *= w[:,1]
            gradw[:,1] *= w[:,0]
            grad2w[:,0] *= w[:,1]
            grad2w[:,1] *= w[:,0]
        elif self.ndim == 3:
            gradw[:,0] *= w[:,1]*w[:,2]
            gradw[:,1] *= w[:,0]*w[:,2]
            gradw[:,2] *= w[:,0]*w[:,1]
            grad2w[:,0] *= w[:,1]*w[:,2]
            grad2w[:,1] *= w[:,0]*w[:,2]
            grad2w[:,2] *= w[:,0]*w[:,1]
        w = np.prod(w, axis=1)
        return indices, w, gradw, grad2w