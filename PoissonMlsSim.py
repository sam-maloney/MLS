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
    See MlsSim documentation for general MLS attributes and methods.
    
    Attributes
    ----------
    g : function object
        Function defining the solution Dirichlet values along the boundary.
    method : string, optional
        Name of method used for assembling the stiffness matrix.
    isBoundaryNode : numpy.ndarray, dtype='bool'
        Information on whether a given node is on the Dirichlet boundary.
    nBoundaryNodes : int
        Numer of nodes on the Dirichlet boundary.
    boundaryValues : numpy.ndarray, dtype='float64'
        Stored values of g() evaluated at the boundary nodes.
    
    Methods
    -------
    selectMethod(self, method, quadrature)
        Register the 'self.assembleStiffnesMatrix' method.
    defineSupport(self, point)
        Find nodes within support of a given evaluation point.
    solve(self, preconditioner=None, **kwargs):
        Solve for the approximate solution using an iterative solver.
        
    """
   
    def __init__(self, N, g, method='galerkin', quadrature='uniform',
                 **kwargs):
        """Construct PoissonMlsSim by extending MlsSim constructor
    
        Parameters
        ----------
        N : integer
            Number of grid cells along one dimension. Must be greater than 0.
        g : function object
            Function defining the solution Dirichlet values along the boundary.
            The object must take an nx2 numpy.ndarray of points and return a
            1D numpy.ndarray of size n for the function values at those points.
        method : string, optional
            Method used for assembling the stiffness matrix.
            Must be either 'galerkin' or 'collocation'. Default is 'galerkin'.
        quadrature : string, optional
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'. Default is 'uniform'.
        **kwargs
            Keyword arguments to be passed to base MlsSim class constructor.
            See the MlsSim class documentation for details.
        
        Returns
        -------
        None.
    
        """
        super().__init__(N, ndim=2, **kwargs)
        self.nNodes = (N+1)*(N+1)
        self.nodes = ( np.indices((N+1, N+1), dtype='float64')
                       .reshape(2,-1).T ) / N
        self.isBoundaryNode = np.any(np.mod(self.nodes, 1) == 0, axis=1)
        self.nBoundaryNodes = np.count_nonzero(self.isBoundaryNode)
        self.boundaryValues = g(self.nodes[self.isBoundaryNode]) \
                                .round(decimals=14)
        self.g = g
        self.selectMethod(method, quadrature)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}, {self.g.__name__}, " \
               f"'{self.method}', '{self.quadrature}', Nquad={self.Nquad}, " \
               f"support={repr(self.support)}, form='{self.form}', " \
               f"basis='{self.basis.name}')"
    
    def selectMethod(self, method='galerkin', quadrature='uniform'):
        """Register the 'self.assembleStiffnesMatrix' method.
        
        Parameters
        ----------
        method : string
            Method used for assembling the stiffness matrix.
            Must be either 'galerkin' or 'collocation'. Default is 'galerkin'.
        quadrature : string
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'. Default is 'uniform'.

        Returns
        -------
        None.

        """
        self.method = method.lower()
        if self.method == 'galerkin':
            self.assembleStiffnessMatrix = self.assembleGalerkinStiffnessMatrix
            self.generateQuadraturePoints(quadrature)
            self.b = np.concatenate((np.zeros(self.nNodes,dtype='float64'),
                                     self.boundaryValues))
        elif self.method == 'collocation':
            self.assembleStiffnessMatrix = self.assembleCollocationStiffnessMatrix
            self.b = np.zeros(self.nNodes,dtype='float64')
            self.b[self.isBoundaryNode] = self.boundaryValues
        else:
            print(f"Error: unkown assembly method '{method}'. "
                  f"Must be one of 'galerkin' or 'collocation'.")
    
    def assembleGalerkinStiffnessMatrix(self):
        """Assemble the Galerkin system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        nMaxEntries = int((self.nNodes * self.support.volume)**2 * self.nQuads)
        data = np.zeros(nMaxEntries, dtype='float64')
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        # build matrix for interior nodes
        index = 0
        for iQ, quad in enumerate(self.quads):
            indices, gradphis = self.dphi(quad)[0:3:2]
            nEntries = len(indices)**2
            data[index:index+nEntries] = np.ravel(gradphis @ gradphis.T)
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        # assemble the triplets into the sparse stiffness matrix
        self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                                shape=(self.nNodes, self.nNodes) )
        self.K *= self.quadWeight
        # pre-allocate arrays for additional stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        nMaxEntries = int( (self.nNodes * self.support.volume)**2
                          * self.nBoundaryNodes )
        data = np.zeros(nMaxEntries, dtype='float64')
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
            indices, phis = self.phi(node)
            nEntries = len(indices)
            data[index:index+nEntries] = -1.0*phis
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
        # pre-allocate arrays for constructing stiffness matrix
        # this is the maximum possibly required size; not all will be used
        nMaxEntries = int(self.nNodes**2 * self.support.volume)
        data = np.empty(nMaxEntries, dtype='float64')
        indices = np.empty(nMaxEntries, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes):
            indptr[iN] = index
            if (self.isBoundaryNode[iN]):
                inds, phis = self.phi(node)
                nEntries = len(inds)
                data[index:index+nEntries] = phis
            else:
                inds, d2phis = self.d2phi(node)[0:3:2]
                nEntries = len(inds)
                data[index:index+nEntries] = d2phis.sum(axis=1)
            indices[index:index+nEntries] = inds
            index += nEntries
        # print(f"{index}/{nMaxEntries} used for spatial discretization")
        indptr[-1] = index
        self.K = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                                shape=(self.nNodes, self.nNodes) )
    
    def precondition(self, preconditioner=None, M=None):
        """Generate and/or store the preconditioning matrix M.

        Parameters
        ----------
        preconditioner : {string, None}, optional
            Which preconditioning method to use.
            Must be one of 'jacobi', 'ilu', or None. The default is None.
        M : {scipy.sparse.linalg.LinearOperator, None}, optional
            Used to directly specifiy the linear operator to be used.
            Only used if preconditioner==None, otherwise ignored.
            The default is None.

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
    
    def solve(self, preconditioner=None, **kwargs):
        """Solve for the approximate solution using an iterative solver.

        Parameters
        ----------
        preconditioner : {string, None}, optional
            Which preconditioning method to use.
            See mls.MlsSim.precontion() for details. The default is None.
        **kwargs
            Keyword arguments to be passed to the scipy solver routine.
            See the scipy.sparse.linalg.lgmres documentation for details.

        Returns
        -------
        None.

        """
        if "M" not in kwargs:
            kwargs["M"] = None
        self.precondition(preconditioner, kwargs["M"])
        kwargs["M"] = self.M
        uTmp, self.info = sp_la.lgmres(self.K, self.b, **kwargs)
        if (self.info != 0):
            print(f'solution failed with error code: {self.info}')
        # uTmp = sp_la.spsolve(self.K, self.b) # direct solver for testing
        # reconstruct final u vector from shape functions
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.nodes):
            indices, phis = self.phi(node)
            self.u[iN] = uTmp[indices] @ phis
            
    def cond(self, order=2, preconditioned=True):
        """Computes the condition number of the stiffness matrix K.
        
        Parameters
        ----------
        order : {int, inf, -inf, ‘fro’}, optional
            Order of the norm. inf means numpy’s inf object. The default is 2.
        preconditioned : bool, optional
            Whether to compute the condition number with the preconditioning
            operation applied to the stiffness matrix. The default is True.

        Returns
        -------
        float
            The condition number of the matrix.

        """
        if preconditioned:
            return super().cond(self.K, self.M, order)
        else:
            return super().cond(self.K, None, order)
