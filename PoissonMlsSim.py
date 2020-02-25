#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:20:15 2020

@author: Sam Maloney
"""

import mls
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la


class PoissonMlsSim(mls.MlsSim):
    """Class for solving Poisson problem using meshfree moving least squares.
    
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
            self.weightFunction = mls.CubicSpline()
        elif form.lower() == 'quartic':
            self.weightFunction = mls.QuarticSpline()
        elif form.lower() == 'gaussian':
            self.weightFunction = mls.Gaussian()
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
            phi, gradphi = mls.shapeFunctions1(quad, self.nodes[indices],
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
            phi = mls.shapeFunctions0(node, self.nodes[indices],
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
                phi = mls.shapeFunctions0(node, self.nodes[inds],
                                          self.weightFunction, self.support)
                data[index:index+nEntries] = phi
            else:
                phi, d2phi = mls.shapeFunctions2(node, self.nodes[inds],
                                                 self.weightFunction, self.support)
                data[index:index+nEntries] = d2phi.sum(axis=1)
            index += nEntries
        indptr[-1] = index
        self.K = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                                shape=(self.nNodes, self.nNodes) )
    
    def solve(self, preconditioner=None, **kwargs):
        """Solve for the approximate solution using an iterative solver.

        Parameters
        ----------
        preconditioner : {string, None}, optional
            Which preconditioning method to use.
            See mls.MlsSim.precontion() for details. The default is None.
        **kwargs
            Keyword arguments to be passed to the scipy solver routine.
            See the scipy.spare.linalg.lgmres documentation for details.

        Returns
        -------
        None.

        """
        if "M" not in kwargs:
            kwargs["M"] = None
        super(PoissonMlsSim, self).precondition(preconditioner, kwargs["M"])
        uTmp, self.info = sp_la.lgmres(self.K, self.b, **kwargs)
        if (self.info != 0):
            print(f'solution failed with error code: {self.info}')
        # uTmp = sp_la.spsolve(self.K, self.b) # direct solver for testing
        # reconstruct final u vector from shape functions
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.nodes):
            indices = self.defineSupport(node)
            phi = mls.shapeFunctions0(node, self.nodes[indices],
                                      self.weightFunction, self.support)
            self.u[iN] = uTmp[indices]@phi
