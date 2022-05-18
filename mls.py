#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:20:15 2020

@author: Sam Maloney

Classes
-------
Support : metaclass=ABCMeta

CircularSupport : Support

RectangularSupport : Support


Basis : metaclass=ABCMeta

LinearBasis : Basis

QuadraticBasis : Basis


MlsSim : metaclass=ABCMeta

"""

import scipy
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

import supports
import bases
import weightfunctions

from abc import ABCMeta, abstractmethod

# from WeightFunction import *


class MlsSim(metaclass=ABCMeta):
    """Class for meshless moving least squares (MLS) method.

    Attributes
    ----------
    N : int
        Number of grid cells along one dimension. Must be greater than 0.
    Nquad : int
        Number of quadrature points in each grid cell along one dimension.
    ndim : int
        The number of spatial dimensions to simulate.
    dx : float
        Grid spacing of underlying uniform grid
    support : Support
        Object defining shape and size of local shape function domains.
    form : string
        Name of the kernel weighting function.
    weightFunction : WeightFunction
        Object defining the kernel weighting function computations.
    basis : Basis
        Object defining the polynomial basis defining the MLS aproximation.
    nNodes : int
        Number of unique nodal points in the simulation domain.
    nodes : numpy.ndarray, dtype='float64', shape=(..., ndim)
        Coordinates of all nodes in the simulation.

    Attributes for Galerkin Assembly
    --------------------------------
    quadrature : string
        Distribution of quadrature points in each cell.
    nQuads : int
        Number of quadrature points in the simulation domain.
    quads : numpy.ndarray, dtype='float64', shape=(nQuads, ndim)
         Coordinates of all quadrature points in the simulation.
    quadWeights : numpy.ndarray, dtype='float64', shape=(nQuads,)
        Relative weighting of each quadrature point in the numerical sum.

    Methods
    -------
    generateQuadraturePoints(self, quadrature):
        Compute array of quadrature points for Galerkin integration.
    selectWeightFunction(self, form):
        Register the 'self.weightFunction' object to the correct kernel.
    selectSupport(self, support):
        Register the 'self.support' object to the correct shape.
    selectBasis(self, name):
        Register the 'self.basis' object to the correct basis set.
    phi(self, point):
        Compute shape function value evaluated at point.
    dphi(self, point):
        Compute shape function value and gradient evaluated at point.
    d2phi(self, point):
        Compute shape function value and laplacian evaluated at point.
    uNodes(self):
        Return the set of nodes on which the solution is computed.
    solve(self, *args, **kwargs):
        Compute the true approximate solution.
    cond(self, A, M=None, order=2):
        Compute the condition number of the matrix A preconditioned by M.

    """

    def __init__(self, N, ndim=2, Nquad=2, support=-1, form='cubic',
                 basis='linear', **kwargs):
        """Initialize shared attributes of MLS simulation classes

        Parameters
        ----------
        N : integer
            Number of grid cells along one dimension. Must be greater than 0.
        Nquad : integer, optional
            Number of quadrature points in each grid cell along one dimension.
            Must be > 0. The default is 2.
        support : {float, (string, float), Support}
            Support size, or (shape, size) pair of the shape function domains,
            or Support object. If present, the shape string must be one of
            'circular' or 'rectangular'. The default is -1.
        form : {string, WeightFunction}, optional
            Form of the spline to be used for the kernel weighting function.
            If a string, must be either 'cubic', 'quartic', or 'gaussian'.
            Otherwise a Weightfunction object can be directly specified.
            The default is 'cubic'.
        basis : {string, Basis}, optional
            Complete polynomial basis defining the MLS aproximation.
            If a string, must be either 'linear' or 'quadratic'.
            The default is 'linear'.
        ndim : int, optional
            The number of spatial dimensions to simulate. The default is 2.
        **kwargs
            Extraneous keyword arguments passed by subclass constructor.

        Returns
        -------
        None.

        """
        self.N = N
        self.Nquad = Nquad
        self.ndim = ndim
        self.dx = 1/N
        self.selectWeightFunction(form)
        self.selectSupport(support)
        self.selectBasis(basis)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}," \
               f"{self.Nquad},{self.support.size*self.N},'{self.form}',"

    def generateQuadraturePoints(self, quadrature):
        """Compute array of quadrature points for Galerkin integration.

        Parameters
        ----------
        quadrature : string
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'.

        Returns
        -------
        None.

        """
        self.quadrature = quadrature.lower()
        if quadrature.lower() not in ['uniform', 'gaussian']:
            print(f"Error: bad quadrature distribution of '{quadrature}'. "
                  f"Must be either 'uniform' or 'gaussian'. "
                  f"Defaulting to 'gaussian'.")
            self.quadrature = 'gaussian'
        if (self.Nquad <= 0) and not float(self.Nquad).is_integer():
            raise SystemExit(f"Bad Nquad value of '{self.Nquad}'. "
                             f"Must be an integer greater than 0.")
        NquadN = self.Nquad*self.N
        if quadrature.lower() == 'uniform' or self.Nquad == 1:
            self.quads = ( np.indices(np.repeat(NquadN, self.ndim),
                           dtype='float64').T.reshape(-1,self.ndim) ) \
                         / NquadN + 1.0/(2*NquadN)
            self.quadWeights = np.repeat(1.0/len(self.quads), len(self.quads))
        elif quadrature.lower() == 'gaussian':
            offsets, weights = scipy.special.roots_legendre(self.Nquad)
            offsets /= (2*self.N)
            weights /= (2*self.N)
            self.quads = ( np.indices(np.repeat(self.N, self.ndim),
                dtype='float64').T.reshape(-1, self.ndim) + 0.5 ) / self.N
            self.quadWeights = np.repeat(1., len(self.quads))
            for i in range(self.ndim):
                self.quads = np.concatenate( [self.quads +
                    offset*np.eye(self.ndim)[i] for offset in offsets] )
                self.quadWeights = np.concatenate(
                    [self.quadWeights * weight for weight in weights] )
        self.nQuads = len(self.quads)

    def selectWeightFunction(self, form):
        """Register the 'self.weightFunction' object to the correct kernel.

        Parameters
        ----------
        form : {string, WeightFunction}
            Form of the kernel weighting function, or WeightFunction object.
            If a string, it must be one of 'cubic', 'quartic', or 'gaussian'.

        Returns
        -------
        None.

        """
        if isinstance(form, weightfunctions.WeightFunction):
            self.weightFunction = form
            self.form = form.form
            return
        self.form = form.lower()
        if self.form == 'linear':
            self.weightFunction = weightfunctions.LinearSpline()
        elif self.form == 'quadratic':
            self.weightFunction = weightfunctions.QuadraticSpline()
        elif self.form == 'cubic':
            self.weightFunction = weightfunctions.CubicSpline()
        elif self.form == 'quartic':
            self.weightFunction = weightfunctions.QuarticSpline()
        elif self.form == 'quintic':
            self.weightFunction = weightfunctions.QuinticSpline()
        elif self.form == 'gaussian':
            self.weightFunction = weightfunctions.Gaussian()
        elif self.form == 'bump':
            self.weightFunction = weightfunctions.Bump()
        else:
            raise SystemExit(f"Unkown spline form '{form}'. Must be one of "
                             f"'cubic', 'quartic', or 'gaussian' or an "
                             f"obect derived from mls.WeightFunction.")

    def selectSupport(self, support):
        """Register the 'self.support' object to the correct shape.

        Parameters
        ----------
        support : {float, (string, float), Support}
            Support size or (shape, size) pair of the shape function, or
            Support object. If present, the shape string must be one of
            'circular' or 'rectangular'.

        Returns
        -------
        None.

        """
        if isinstance(support, supports.Support):
            self.support = support
            return
        if type(support) in [int, float]:
            if support > 0:
                self.support = supports.CircularSupport(self, support/self.N)
            else: # if support size is negative, use a default
                self.support = supports.CircularSupport(self, 1.8/self.N)
            return
        size = support[1] / self.N
        if size <= 0: # if support size is negative, use a default
            size = 1.8/self.N
        if support[0].lower() == 'circular':
            self.support = supports.CircularSupport(self, size)
        elif support[0].lower() == 'rectangular':
            self.support = supports.RectangularSupport(self, size)
        else:
            raise SystemExit(f"Unkown support {support}. Must be a numeric "
                             f"size, one of ('circular', size) or "
                             f"('rectangular', size), or an obect derived "
                             f"from mls.Support.")

    def selectBasis(self, name):
        """Register the 'self.basis' object to the correct basis set.

        Parameters
        ----------
        name : {string, Basis}
            Name of the basis to be used for the shape functions, or Basis
            object. If a string, it must be either 'linear' or 'quadratic'.

        Returns
        -------
        None.

        """
        if isinstance(name, bases.Basis):
            self.basis = name
            return
        name = name.lower()
        if name == 'linear':
            self.basis = bases.LinearBasis(self.ndim)
        elif name == 'quadratic':
            self.basis = bases.QuadraticBasis(self.ndim)
        else:
            raise SystemExit(f"Unkown basis name '{name}'. Must be either "
                "'linear' or 'quadratic' or an object derived from mls.Basis.")

    def phi(self, point):
        """Compute shape function value at given point.
        Does not compute any derivatives.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        phi : numpy.ndarray, dtype='float64', shape=(n,)
            Values of phi for all nodes in self.nodes[indices].

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w = self.support.w(point)
        p = self.basis(self.nodes[indices])
        A = w*p.T@p
        # --------------------------------------
        #      compute vector c(x) and phi
        # --------------------------------------
        # A(x)c(x) = p(x)
        # Backward substitution for c(x) using LU factorization for A
        p_x = self.basis(point)[0]
        lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
        c = la.lu_solve((lu, piv), p_x, overwrite_b=True, check_finite=False)
        phi = c @ p.T * w
        return indices, phi

    def dphi(self, point):
        """Compute shape function value and gradient at given point.
        Does not compute second derivatives.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        phi : numpy.ndarray, dtype='float64', shape=(n,)
            Values of phi for all n nodes in self.nodes[indices].
        gradphi : numpy.ndarray, dtype='float64', shape=(n,ndim)
            Gradients of phi for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dx1,dy1,dz1],[dx2,dy2,dz2]...])

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, gradw = self.support.dw(point)
        p = self.basis(self.nodes[indices])
        A = w*p.T@p
        dA = [gradw[:,i]*p.T@p for i in range(self.ndim)]
        # --------------------------------------
        #         compute matrix c
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), ndim times for c_k(x)
        # Using LU factorization for A
        p_x = self.basis(point)[0]
        lu, piv = la.lu_factor(A, check_finite=False)
        c = np.empty((self.ndim + 1, self.basis.size))
        c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
        for i in range(self.ndim):
            c[i+1] = la.lu_solve( (lu, piv),
                                  (self.basis.dp(point)[i] - dA[i]@c[0]),
                                  check_finite=False )
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        cpi = c[0] @ p.T
        phi = cpi * w
        gradphi = ( c[1 : self.ndim + 1]@p.T*w + cpi*gradw.T).T
        return indices, phi, gradphi

    def d2phi(self, point):
        """Compute shape function value and laplacian at given point.
        Does not compute the 1st order gradient.

        Parameters
        ----------
        point : numpy.ndarray, dtype='float64', shape=(ndim,)
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.ndarray, dtype='uint32', shape=(n,)
            Indices of nodes with non-zero support at evaluation point.
        phi : numpy.ndarray, dtype='float64', shape=(n,)
            Values of phi for all n nodes in self.nodes[indices].
        grad2phi : numpy.ndarray, dtype='float64', shape=(n,ndim)
            2nd derivatives of phi for all n nodes in self.nodes[indices].
            Has the form numpy.array([[dxx1,dyy1,dzz1],[dxx2,dyy2,dzz2]...]).

        """
        # --------------------------------------
        #     compute the moment matrix A(x)
        # --------------------------------------
        indices, w, gradw, grad2w = self.support.d2w(point)
        p = self.basis(self.nodes[indices])
        A = w*p.T@p
        dA = [gradw[:,i]*p.T@p for i in range(self.ndim)]
        d2A = [grad2w[:,i]*p.T@p for i in range(self.ndim)]
        # --------------------------------------
        #         compute  matrix c(x)
        # --------------------------------------
        # A(x)c(x)   = p(x)
        # A(x)c_k(x) = b_k(x)
        # Backward substitutions, once for c(x), ndim times for c_k(x)
        # and ndim times for c_kk(x), using LU factorization for A
        p_x = self.basis(point)[0]
        lu, piv = la.lu_factor(A, check_finite=False)
        c = np.empty((2*self.ndim + 1, self.basis.size))
        c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
        for i in range(self.ndim):
            c[i+1] = la.lu_solve( (lu, piv),
                                  (self.basis.dp(point)[i] - dA[i]@c[0]),
                                  check_finite=False )
            c[i+1+self.ndim] = la.lu_solve( (lu, piv),
                (self.basis.d2p(point)[i] - 2.0*dA[i]@c[i+1] - d2A[i]@c[0]),
                check_finite=False )
        # --------------------------------------
        #       compute phi and gradphi
        # --------------------------------------
        cpi = c[0] @ p.T
        phi = cpi * w
        grad2phi = ( c[self.ndim + 1 : 2*self.ndim + 1]@p.T*w +
                     2.0*c[1 : self.ndim + 1]@p.T*gradw.T +
                     cpi*grad2w.T ).T
        return indices, phi, grad2phi

    def uNodes(self):
        """Return the set of nodes on which the solution is computed.

        Returns
        -------
        nx2 numpy.ndarray, dtype='float64'
            Default implementation returns the full set of self.nodes.

        """
        return self.nodes

    # @abstractmethod
    def solve(self, *args, **kwargs):
        """ABSTRACT METHOD. Compute the true approximate solution."""
        pass

    def cond(self, A, M=None, order=2):
        """Compute the condition number of the matrix A preconditioned by M.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            The matrix for which to compute the condition number.
        M : scipy.sparse.linalg.LinearOperator, optional
            Preconditioner for matrix A in order to compute the condition
            number of the preconditioned system. The default is None.
        order : {int, inf, -inf, ‘fro’}, optional
            Order of the norm. inf means numpy’s inf object. The default is 2.

        Returns
        -------
        c : float
            The condition number of the matrix.

        """
        if M != None:
            A = M @ A.A
        if order == 2:
            LM = sp_la.svds(A, 1, which='LM', return_singular_vectors=False)
            SM = sp_la.svds(A, 1, which='SM', return_singular_vectors=False)
            c = LM[0]/SM[0]
        else:
            if sp.issparse(A):
                c = sp_la.norm(A, order) * sp_la.norm(sp_la.inv(A), order)
            else: # A is dense
                c = la.norm(A, order) * la.norm(la.inv(A), order)
        return c
