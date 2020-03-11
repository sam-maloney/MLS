#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:20:15 2020

@author: Sam Maloney
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

from abc import ABCMeta, abstractmethod


def shapeFunctions0(point, nodes, weightFunction, support):
    """Compute shape function at quad point for all nodes in its support.
    Basis used is linear basis pT = [1 x y].
    Computes the shape function value only (no derivatives).

    Parameters
    ----------
    point : numpy.array([x,y], dtype='float64')
        Coordinate of evaluation point.
    nodes : nx2 numpy.ndarray, dtype='float64'
        Coordinates of nodes within support of given evaluation point.

    Returns
    -------
    phi : numpy.array([...], dtype='float64')
        Values of phi for all nodes in indices.

    """
    # --------------------------------------
    #     compute the moment matrix A(x)
    # --------------------------------------
    distances = la.norm(point - nodes, axis=-1)/support
    w = weightFunction.w(distances)
    p = np.hstack((np.ones((len(nodes),1)), nodes))
    A = w*p.T@p
    # --------------------------------------
    #      compute  matrix c(x) and phi
    # --------------------------------------
    # A(x)c(x) = p(x)
    # Backward substitution for c(x) using LU factorization for A
    p_x = np.concatenate(([1.0], point))
    lu, piv = la.lu_factor(A, overwrite_a=True, check_finite=False)
    c = la.lu_solve((lu, piv), p_x, overwrite_b=True, check_finite=False)
    phi = c @ p.T * w
    return phi

def shapeFunctions1(point, nodes, weightFunction, support):
    """Compute shape function at quad point for all nodes in its support.
    Basis used is linear basis pT = [1 x y].
    Computes the shape function value and its gradient.

    Parameters
    ----------
    point : numpy.array([x,y], dtype='float64')
        Coordinate of evaluation point.
    nodes : nx2 numpy.ndarray, dtype='float64'
        Coordinates of nodes within support of given evaluation point.

    Returns
    -------
    phi : numpy.array([...], dtype='float64')
        Values of phi for all nodes in indices.
    gradphi : nx2 numpy.ndarray, dtype='float64'
        Gradients of phi for all nodes in indices. [[dx1,dy1],[dx2,dy2]...]

    """
    # --------------------------------------
    #     compute the moment matrix A(x)
    # --------------------------------------
    displacement = (point - nodes)/support
    distance = np.array(la.norm(displacement, axis=-1))
    w, dwdr = weightFunction.dw(distance)
    i0 = distance > 1e-14
    gradr = np.full(nodes.shape, np.sqrt(0.5)/support, dtype='float64')
    gradr[i0] = displacement[i0] / \
                (distance[i0]*support).reshape((-1,1))
    gradw = dwdr.reshape((-1,1)) * gradr
    p = np.hstack((np.ones((len(nodes),1)), nodes))
    A = w*p.T@p
    dAdx = gradw[:,0]*p.T@p
    dAdy = gradw[:,1]*p.T@p
    # --------------------------------------
    #         compute  matrix c(x)
    # --------------------------------------
    # A(x)c(x)   = p(x)
    # A(x)c_k(x) = b_k(x)
    # Backward substitutions, once for c(x), twice for c_k(x) k=1,2
    # Using LU factorization for A
    p_x = np.concatenate(([1.0], point))
    lu, piv = la.lu_factor(A, check_finite=False)
    c = np.empty((3,3), dtype='float64')
    c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
    c[1] = la.lu_solve((lu, piv),([0,1,0] - dAdx@c[0]), check_finite=False)
    c[2] = la.lu_solve((lu, piv),([0,0,1] - dAdy@c[0]), check_finite=False)
    # --------------------shapeFunctions1------------------
    #       compute phi and gradphi
    # --------------------------------------
    cpi = c[0] @ p.T
    phi = cpi * w
    gradphi = ( c[1:3]@p.T*w + cpi*gradw.T).T
    return phi, gradphi

def shapeFunctions2(point, nodes, weightFunction, support):
    """Compute shape function at quad point for all nodes in its support.
    Basis used is linear basis pT = [1 x y].
    Computes the shape function value and its 2nd derivatives.

    Parameters
    ----------
    point : numpy.array([x,y], dtype='float64')
        Coordinate of evaluation point.
    nodes : nx2 numpy.ndarray, dtype='float64'
        Coordinates of nodes within support of given evaluation point.

    Returns
    -------
    phi : numpy.array([...], dtype='float64')
        Values of phi for all nodes in indices.
    grad2phi : nx2 numpy.ndarray, dtype='float64'
        2nd derivatives of phi for all nodes in indices.
        [[dxx1,dyy1],[dxx2,dyy2]...]

    """
    # --------------------------------------
    #     compute the moment matrix A(x)
    # --------------------------------------
    displacement = (point - nodes)/support
    distance = np.array(la.norm(displacement, axis=-1))
    w, dwdr, d2wdr2 = weightFunction.dw2(distance)
    i0 = distance > 1e-14
    gradr = np.full(nodes.shape, np.sqrt(0.5)/support, dtype='float64')
    gradr[i0] = displacement[i0] / \
                (distance[i0]*support).reshape((-1,1))
    gradw = dwdr.reshape((-1,1)) * gradr
    grad2w = d2wdr2.reshape((-1,1)) * gradr*gradr
    p = np.hstack((np.ones((len(nodes),1)), nodes))
    A = w*p.T@p
    dAdx = gradw[:,0]*p.T@p
    dAdy = gradw[:,1]*p.T@p
    d2Adx2 = grad2w[:,0]*p.T@p
    d2Ady2 = grad2w[:,1]*p.T@p
    # --------------------------------------
    #         compute  matrix c(x)
    # --------------------------------------
    # A(x)c(x)   = p(x)
    # A(x)c_k(x) = b_k(x)
    # Backward substitutions, once for c(x), twice for c_k(x) k=1,2
    # and twice for c_kk(x) k=1,2, using LU factorization for A
    p_x = np.concatenate(([1.0], point))
    lu, piv = la.lu_factor(A, check_finite=False)
    c = np.empty((5,3), dtype='float64')
    c[0] = la.lu_solve((lu, piv), p_x, check_finite=False)
    c[1] = la.lu_solve((lu, piv),([0,1,0] - dAdx@c[0]), check_finite=False)
    c[2] = la.lu_solve((lu, piv),([0,0,1] - dAdy@c[0]), check_finite=False)
    c[3] = la.lu_solve((lu, piv),(-2.0*dAdy@c[1] - d2Adx2@c[0]), check_finite=False)
    c[4] = la.lu_solve((lu, piv),(-2.0*dAdy@c[2] - d2Ady2@c[0]), check_finite=False)
    # --------------------------------------
    #       compute phi and gradphi
    # --------------------------------------
    cpi = c[0] @ p.T
    phi = cpi * w
    grad2phi = ( c[3:5]@p.T*w + 2.0*c[1:3]@p.T*gradw.T + cpi*grad2w.T ).T
    return phi, grad2phi


class shapeFunction(object):
    def phi(self, point):
        pass
    
    def dphi(self, point):
        pass
    
    def d2phi(self, point):
        pass


class WeightFunction(metaclass=ABCMeta):
    @property
    @abstractmethod
    def form(self): pass

    @abstractmethod
    def w(self, r): pass
    
    @abstractmethod
    def dw(self, r): pass
    
    @abstractmethod
    def dw2(self, r): pass

class CubicSpline(WeightFunction):
    @property
    def form(self):
        return 'cubic'
    
    def w(self, r):
        """Compute cubic spline function value.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the cubic spline function at the given distances.

        """
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size, dtype='float64')
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            w[i0] = 2.0/3.0 - 4.0*r2 + 4.0*r3
        if i1.any():
            r1 = r[i1]
            r2 = r1*r1
            r3 = r2*r1
            w[i1] = 4.0/3.0 - 4.0*r1 + 4.0*r2 - 4.0/3.0*r3
        return w
    
    def dw(self, r):
        """Compute cubic spline function and its radial derivative.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the cubic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.

        """
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            w[i0] = 2.0/3.0 - 4.0*r2 + 4.0*r3
            dwdr[i0] = -8.0*r1 + 12.0*r2
        if i1.any():
            r1 = r[i1]
            r2 = r1*r1
            r3 = r2*r1
            w[i1] = 4.0/3.0 - 4.0*r1 + 4.0*r2 - 4.0/3.0*r3
            dwdr[i1] = -4.0 + 8.0*r1 - 4.0*r2
        return w, dwdr
    
    def dw2(self, r):
        """Compute cubic spline function and its radial derivatives.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the cubic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.array([...], dtype='float64')
            Values of the 2nd order radial derivative at the given distances.

        """
        i0 = r < 0.5
        i1 = np.logical_xor(r < 1, i0)
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            w[i0] = 2.0/3.0 - 4.0*r2 + 4.0*r3
            dwdr[i0] = -8.0*r1 + 12.0*r2
            d2wdr2[i0] = -8.0 + 24.0*r1
        if i1.any():
            r1 = r[i1]
            r2 = r1*r1
            r3 = r2*r1
            w[i1] = 4.0/3.0 - 4.0*r1 + 4.0*r2 - 4.0/3.0*r3
            dwdr[i1] = -4.0 + 8.0*r1 - 4.0*r2
            d2wdr2[i1] = 8.0 - 8.0*r1
        return w, dwdr, d2wdr2

class QuarticSpline(WeightFunction):
    @property
    def form(self):
        return 'quartic'

    def w(self, r):
        """Compute quartic spline function values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the quartic spline function at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            r4 = r2*r2
            w[i0] = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
        return w
    
    def dw(self, r):
        """Compute quartic spline function and radial derivative values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the quartic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            r4 = r2*r2
            w[i0] = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
            dwdr[i0] = -12.0*r1 + 24.0*r2 - 12.0*r3
        return w, dwdr
    
    def dw2(self, r):
        """Compute quartic spline function and radial derivative values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the quartic spline function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.array([...], dtype='float64')
            Values of the 2nd order radial derivative at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1*r1
            r3 = r2*r1
            r4 = r2*r2
            w[i0] = 1.0 - 6.0*r2 + 8.0*r3 - 3.0*r4
            dwdr[i0] = -12.0*r1 + 24.0*r2 - 12.0*r3
            d2wdr2[i0] = -12.0 + 48.0*r1 - 36.0*r2
        return w, dwdr, d2wdr2

class Gaussian(WeightFunction):
    @property
    def form(self):
        return 'gaussian'

    def w(self, r):
        """Compute Gaussian function values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the Gaussian function at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        if i0.any():
            r2 = r[i0]**2
            c1 = np.exp(-9.0)
            w[i0] = (np.exp(-9.0*r2) - c1) / (1.0 - c1)
        return w
    
    def dw(self, r):
        """Compute Gaussian function and radial derivative values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the Gaussian function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1**2
            c1 = np.exp(-9.0)
            c2 = 1.0/(1.0 - c1)
            w[i0] = (np.exp(-9.0*r2) - c1) * c2
            dwdr[i0] = -18.0*r1*np.exp(-9.0*r2) * c2
        return w, dwdr
    
    def dw2(self, r):
        """Compute Gaussian function and radial derivative values.

        Parameters
        ----------
        r : numpy.array([...], dtype='float64')
            Distances from evaluation points to node point.

        Returns
        -------
        w : numpy.array([...], dtype='float64')
            Values of the Gaussian function at the given distances.
        dwdr : numpy.array([...], dtype='float64')
            Values of the radial derivative at the given distances.
        d2wdr2 : numpy.array([...], dtype='float64')
            Values of the 2nd order radial derivative at the given distances.

        """
        i0 = r < 1
        w = np.zeros(r.size, dtype='float64')
        dwdr = w.copy()
        d2wdr2 = w.copy()
        if i0.any():
            r1 = r[i0]
            r2 = r1**2
            c1 = np.exp(-9.0)
            c2 = 1.0/(1.0 - c1)
            w[i0] = (np.exp(-9.0*r2) - c1) * c2
            dwdr[i0] = -18.0*r1*np.exp(-9.0*r2) * c2
            d2wdr2[i0] = 18.0*np.exp(-9.0*r2)*(18.0*r2-1) * c2
        return w, dwdr, d2wdr2


class MlsSim(metaclass=ABCMeta):
    """Class for meshless moving least squares (MLS) method.
    
    Parameters
    ----------
    N : integer
        Number of grid cells along one dimension.
        Must be greater than 0.
    g : function object
        Function defining the solution Dirichlet values along the boundary.
        The object must take an nx2 numpy.ndarray of points and return a
        1D numpy.ndarray of size n for the function values at those points.
    Nquad : integer, optional
        Number of quadrature points in each grid cell along one dimension.
        Must be > 0 and either 1 or 2 if quadrature is 'gaussian'.
        The default is 2.
    support : float, optional
        The size of the shape function support, given as a multiple of the
        grid spacing for the given N if the value is positive.
        Supplying a negative value leads to default support sizes being used,
        namely 1.8 for quartic splines or 1.9 for cubic splines.
        The default is -1.
    form : string, optional
        Form of the spline used for the kernel weighting function.
        Must be either 'cubic', 'quartic', or 'gaussian'.
        The default is 'cubic'.
    method : string, optional
        Method used for assembling the stiffness matrix.
        Must be either 'galerkin' or 'collocation'.
        The default is 'galerkin'.
    quadrature : string, optional
        Distribution of quadrature points in each cell.
        Must be either 'uniform' or 'gaussian'.
        The default is 'gaussian'.
    """
    
    def __init__(self, N, Nquad=2, support=-1, form='cubic', **kwargs):
        self.N = N
        self.nCells = N*N
        self.Nquad = Nquad
        self.selectWeightFunction(form)
        if support > 0:
            self.support = support/N
        else: # if support is negative, set to default grid spacing
            if self.form == 'quartic':
                self.support = 1.8/N
            elif self.form == 'cubic':
                self.support = 1.9/N
            else: # if form is unkown
                self.support = 1.5/N
        self.isBoundaryNode = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.nBoundaryNodes = np.count_nonzero(self.isBoundaryNode)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}," \
               f"{self.Nquad},{self.support*self.N},'{self.form}',"
    
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
                  f"Must be either 'uniform' or 'gaussian'.")
        NquadN = self.Nquad*self.N
        if (self.Nquad <= 0) and not float(self.Nquad).is_integer():
            print(f"Error: bad Nquad value of '{self.Nquad}'. "
                  f"Must be an integer greater than 0.")
            return
        if self.Nquad == 1:
            self.quads = ( np.indices((self.N, self.N), dtype='float64')
                           .T.reshape(-1,2) + 0.5 ) / self.N
        elif quadrature.lower() == 'gaussian' and self.Nquad == 2:
            tmp = ( np.indices((self.N, self.N), dtype='float64')
                           .T.reshape(-1,2) + 0.5 ) / self.N
            offset = 0.5/(np.sqrt(3.0)*self.N)
            self.quads = np.concatenate((
                tmp - offset,
                tmp + offset,
                np.hstack((tmp[:,0:1] + offset, tmp[:,1:2] - offset)),
                np.hstack((tmp[:,0:1] - offset, tmp[:,1:2] + offset)) ))
        elif quadrature.lower() == 'gaussian':
            print(f"Error: bad Nquad value of '{self.Nquad}'. "
                  f"Must be either 1 or 2 for 'gaussian' quadrature.")
            return
        else: ##### Uniform quadrature for Nquad > 1 #####
            self.quads = np.indices((NquadN,NquadN)).T.reshape(-1,2) / \
                (NquadN)+1/(2*NquadN)
        self.quadWeight = 1.0/(NquadN*NquadN)
        self.nQuads = len(self.quads)
    
    def selectWeightFunction(self, form):
        """Register the 'self.weightFunction' object to the correct kernel.
        
        Parameters
        ----------
        form : string
            Form of the spline used for the kernel weighting function.
            Must be either 'cubic', 'quartic', or 'gaussian'.

        Returns
        -------
        None.

        """
        if isinstance(form, WeightFunction):
            self.weightFunction = form
            self.form = form.form
            return
        self.form = form.lower()
        if self.form == 'cubic':
            self.weightFunction = CubicSpline()
        elif self.form == 'quartic':
            self.weightFunction = QuarticSpline()
        elif self.form == 'gaussian':
            self.weightFunction = Gaussian()
        else:
            print(f"Error: unkown spline form '{form}'. "
                  f"Must be one of 'cubic', 'quartic', or 'gaussian'.")
    
    def defineSupport(self, point):
        """Find nodes within support of a given evaluation point.

        Parameters
        ----------
        point : numpy.array([x,y], dtype='float64')
            Coordinates of given evaluation point.

        Returns
        -------
        indices : numpy.array([...], dtype='uint32')
            Indices of nodes within support of given evaluation point.
            
        """
        distances = la.norm(point - self.nodes, axis=1)
        indices = np.flatnonzero(distances < self.support).astype('uint32')
        return indices
    
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
        """Solve for the approximate solution."""
        pass
    
    def cond(self, A, M=None, order=2):
        """Computes the condition number of the matrix A.
        
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
