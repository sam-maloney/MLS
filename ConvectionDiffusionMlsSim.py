#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:24:15 2020

@author: Sam Maloney
"""

import mls
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la


class ConvectionDiffusionMlsSim(mls.MlsSim):
    """Class for solving convection diffusion equation using meshfree MLS.
    Assumes periodic boundary conditions.
    See MlsSim documentation for general MLS attributes and methods.
    
    Attributes
    ----------
    velocity : np.array([vx,vy], dtype='float64')
        Background velocity of the fluid.
    diffusivity : float
        Diffusion coefficient for the quantity of interest.
    
    Methods
    -------
    selectMethod(self, method, quadrature)
        Register the 'self.assembleStiffnesMatrix' method.
    defineSupport(self, point)
        Find nodes within support of a given evaluation point.
    solve(self, preconditioner=None, **kwargs):
        Solve for the approximate solution using an iterative solver.
    
    """
    
    def __init__(self, N, velocity, diffusivity, **kwargs):
        """Construct ConvectionDiffusionMlsSim by extending MlsSim constructor
    
        Parameters
        ----------
        N : integer
            Number of grid cells along one dimension.
            Must be greater than 0.
        **kwargs
            Keyword arguments to be passed to the base MlsSim class constructor.
            See the MlsSim class documentation for details.
        
        Returns
        -------
        None.
    
        """
        super().__init__(N, **kwargs)
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.selectMethod(kwargs['method'], kwargs['quadrature'])
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}," \
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
        elif method.lower() == 'collocation':
            self.assembleStiffnessMatrix = self.assembleCollocationStiffnessMatrix
        else:
            print(f"Error: unkown assembly method '{method}'. "
                  f"Must be one of 'galerkin' or 'collocation'.")
    
    def computeSpatialDiscretization(self):
        """Assemble the system discretization matrices K, A, M in CSR format.
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative

        Returns
        -------
        None.

        """
        # pre-allocate arrays for discretization matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        nMaxEntriesPerQuad = int((self.nNodes*4*(self.support+0.25/self.N)**2)**2)
        Kdata = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='float64')
        Adata = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='float64')
        Mdata = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='float64')
        row_ind = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='uint32')
        col_ind = np.zeros(self.nQuads * nMaxEntriesPerQuad, dtype='uint32')
        # build matrix for interior nodes
        index = 0
        for iQ, quad in enumerate(self.quads):
            indices = self.defineSupport(quad)
            nEntries = len(indices)**2
            phi, gradphi = mls.shapeFunctions1(quad, self.nodes[indices],
                                           self.weightFunction, self.support)
            Kdata[index:index+nEntries] = np.ravel(gradphi@gradphi.T)
            Adata[index:index+nEntries] = np.ravel(
                np.outer(np.dot(gradphi, self.velocity), phi) )
            Mdata[index:index+nEntries] = np.ravel(np.outer(phi, phi))
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
        K_inds = np.flatnonzero(Kdata.round(decimals=14, out=Kdata))
        A_inds = np.flatnonzero(Adata.round(decimals=14, out=Adata))
        M_inds = np.flatnonzero(Mdata.round(decimals=14, out=Mdata))
        # assemble the triplets into the sparse stiffness matrix
        self.K = sp.csr_matrix( (Kdata[K_inds], (row_ind[K_inds], col_ind[K_inds])),
                                shape=(self.nNodes, self.nNodes) )
        self.A = sp.csr_matrix( (Adata[A_inds], (row_ind[A_inds], col_ind[A_inds])),
                                shape=(self.nNodes, self.nNodes) )
        self.M = sp.csr_matrix( (Mdata[M_inds], (row_ind[M_inds], col_ind[M_inds])),
                                shape=(self.nNodes, self.nNodes) )
        self.K *= -self.quadWeight
        self.A *= -self.quadWeight
        self.M *= -self.quadWeight
        
    def defineSupport(self, point):
        distances = la.norm(point - self.nodes, axis=1)
        indices = np.flatnonzero(distances < self.support)
        if point[0] + self.support > 1.0:
            distances = la.norm((point - [1.0, 0.0]) - self.nodes, axis=1)
            indices = np.append(indices, np.flatnonzero(distances < self.support))
        if point[0] - self.support < 0.0:
            distances = la.norm((point + [1.0, 0.0]) - self.nodes, axis=1)
            indices = np.append(indices, np.flatnonzero(distances < self.support))
        if point[1] + self.support > 1.0:
            distances = la.norm((point - [0.0, 1.0]) - self.nodes, axis=1)
            indices = np.append(indices, np.flatnonzero(distances < self.support))
        if point[1] - self.support < 0.0:
            distances = la.norm((point + [0.0, 1.0]) - self.nodes, axis=1)
            indices = np.append(indices, np.flatnonzero(distances < self.support))
        return indices.astype('uint32')
    
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
        self.precondition(preconditioner, kwargs["M"])
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
