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


class WeightFunction(metaclass=ABCMeta):
    @abstractmethod
    def w(self, r): pass
    
    @abstractmethod
    def dw(self, r): pass
    
    @abstractmethod
    def dw2(self, r): pass

class CubicSpline(WeightFunction):
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


class MlsSim(object):
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
    
    def __init__(self, N, g, Nquad=2, support=-1, form='cubic',
                 method='galerkin', quadrature='gaussian'):
        self.N = N
        self.nCells = N*N
        self.nNodes = (N+1)*(N+1)
        self.Nquad = Nquad
        if support > 0:
            self.support = support/N
        else: # if support is negative, set to default grid spacing
            if form.lower() == 'quartic':
                self.support = 1.8/N
            elif form.lower() == 'cubic':
                self.support = 1.9/N
            else: # if form is unkown
                self.support = 1.5/N
        self.nodes = ( np.indices((N+1, N+1), dtype='float64')
                       .T.reshape(-1,2) ) / N
        self.isBoundaryNode = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.nBoundaryNodes = np.count_nonzero(self.isBoundaryNode)
        self.boundaryValues = g(self.nodes[self.isBoundaryNode]) \
                               .round(decimals=14)
        self.g = g
        self.selectWeightFunction(form)
        self.selectMethod(method, quadrature)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N},{self.g}," \
               f"{self.Nquad},{self.support*self.N},'{self.form}'," \
               f"'{self.method}','{self.quadrature}')"
    
    def selectMethod(self, method, quadrature):
        """Register the 'self.assembleStiffnesMatrix' method.
        
        Parameters
        ----------
        method : string
            Method used for assembling the stiffness matrix.
            Must be either 'galerkin' or 'collocation'.
        quadrature : string
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'.

        Returns
        -------
        None.

        """
        self.method = method.lower()
        if method.lower() == 'galerkin':
            self.assembleStiffnessMatrix = self.assembleGalerkinStiffnessMatrix
            self.generateQuadraturePoints(quadrature)
            self.b = np.concatenate((np.zeros(self.nNodes,dtype='float64'),
                                     self.boundaryValues))
        elif method.lower() == 'collocation':
            self.assembleStiffnessMatrix = self.assembleCollocationStiffnessMatrix
            self.b = np.zeros(self.nNodes,dtype='float64')
            self.b[self.isBoundaryNode] = self.boundaryValues
        else:
            print(f"Error: unkown assembly method '{method}'. "
                  f"Must be one of 'galerkin' or 'collocation'.")
    
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
        self.form = form.lower()
        if form.lower() == 'cubic':
            self.weightFunction = CubicSpline()
        elif form.lower() == 'quartic':
            self.weightFunction = QuarticSpline()
        elif form.lower() == 'gaussian':
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
    
    def assembleGalerkinStiffnessMatrix(self):
        """Assemble the Galerkin system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        nMaxEntriesPerQuad = int((self.nNodes*4*(self.support+0.25/self.N)**2)**2)
        data = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='float64')
        row_ind = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='uint32')
        col_ind = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='uint32')
        # build matrix for interior nodes
        index = 0
        for iQ, quad in enumerate(self.quads):
            indices = self.defineSupport(quad)
            nEntries = len(indices)**2
            phi, gradphi = shapeFunctions1(quad, self.nodes[indices],
                                           self.weightFunction, self.support)
            data[index:index+nEntries] = np.ravel(gradphi@gradphi.T)
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        # assemble the triplets into the sparse stiffness matrix
        self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                                shape=(self.nNodes, self.nNodes) )
        self.K *= self.quadWeight
        # apply Dirichlet boundary conditions using Lagrange multiplier method
        data.fill(0.0)
        row_ind.fill(0)
        col_ind.fill(0)
        index = 0
        for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
            indices = self.defineSupport(node)
            nEntries = len(indices)
            phi = shapeFunctions0(node, self.nodes[indices],
                                  self.weightFunction, self.support)
            data[index:index+nEntries] = -1.0*phi
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                           shape=(self.nNodes, self.nBoundaryNodes) )
        G *= -1.0
        self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
    
    def assembleCollocationStiffnessMatrix(self):
        """Assemble the collocation system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate array for indices
        # this is the maximum possibly required size; not all will be used
        nMaxEntriesPerNode = int(self.nNodes*4*(self.support+0.25/self.N)**2)
        data = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='float64')
        indices = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes):
            inds = self.defineSupport(node)
            nEntries = len(inds)
            indptr[iN] = index
            indices[index:index+nEntries] = inds
            if (self.isBoundaryNode[iN]):
                phi = shapeFunctions0(node, self.nodes[inds],
                                      self.weightFunction, self.support)
                data[index:index+nEntries] = phi
            else:
                phi, d2phi = shapeFunctions2(node, self.nodes[inds],
                                             self.weightFunction, self.support)
                data[index:index+nEntries] = d2phi.sum(axis=1)
            index += nEntries
        indptr[-1] = index
        self.K = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                                shape=(self.nNodes, self.nNodes) )
    
    def solve(self, preconditioner=None, x0=None, tol=1e-05, maxiter=1000,
              M=None, callback=None, inner_m=30, outer_k=3, outer_v=None,
              store_outer_Av=True, prepend_outer_v=False, atol=1e-05):
        """Solve for the approximate solution using an iterative solver.

        Parameters
        ----------
        preconditioner : {string, None}
            

        Returns
        -------
        None.

        """
        self.preconditioner = preconditioner
        if preconditioner == None:
            self.M = M
        elif preconditioner.lower() == 'ilu':
            ilu = sp_la.spilu(self.K)
            Mx = lambda x: ilu.solve(x)
            self.M = sp_la.LinearOperator(self.K.shape, Mx)
        elif preconditioner.lower() == 'jacobi':
            if self.method.lower() == 'collocation':
                self.M = sp_la.inv( sp.diags(
                    self.K.diagonal(), format='csc', dtype='float64') )
            else: # if method == 'galerkin'
                self.M = M
                print("Error: 'jacobi' preconditioner not compatible with "
                      "'galerkin' assembly method. Use 'ilu' or None instead."
                      " Defaulting to None.")
        # uTmp, self.info = sp_la.bicgstab(self.K, self.b, x0, tol, maxiter,
        #                                  self.M, callback, atol)
        uTmp, self.info = sp_la.lgmres(self.K, self.b,
            x0, tol, maxiter, self.M, callback, inner_m, outer_k, outer_v,
            store_outer_Av, prepend_outer_v, atol)
        uTmp = sp_la.spsolve(self.K, self.b) # direct solver for testing
        if (self.info != 0):
            print(f'solution failed with error code: {self.info}')
        # reconstruct final u vector from shape functions
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.nodes):
            indices = self.defineSupport(node)
            phi = shapeFunctions0(node, self.nodes[indices],
                                  self.weightFunction, self.support)
            self.u[iN] = uTmp[indices]@phi
    
    def cond(self, ord=2, preconditioned=True):
        """Computes the condition number of the stiffness matrix K.
        
        Parameters
        ----------
        ord : {int, inf, -inf, ‘fro’}, optional
            Order of the norm. inf means numpy’s inf object. The default is 2.
        preconditioned : bool, optional
            Whether to compute the condition number with the preconditioning
            operation applied to the stiffness matrix. The default is True.

        Returns
        -------
        c : float
            The condition number of the matrix.

        """
        if preconditioned and self.M != None:
            A = self.M @ self.K.A
        else:
            A = self.K
        if ord == 2:
            LM = sp_la.svds(A, 1, which='LM', return_singular_vectors=False)
            SM = sp_la.svds(A, 1, which='SM', return_singular_vectors=False)
            c = LM[0]/SM[0]
        else:
            if sp.issparse(A):
                c = sp_la.norm(A, ord) * sp_la.norm(sp_la.inv(A), ord)
            else: # A is dense
                c = la.norm(A, ord) * la.norm(la.inv(A), ord)
        return c
