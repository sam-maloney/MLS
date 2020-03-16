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
    
    def __init__(self, N, dt, u0, velocity, diffusivity, **kwargs):
        """Construct ConvectionDiffusionMlsSim by extending MlsSim constructor
    
        Parameters
        ----------
        N : integer
            Number of grid cells along one dimension.
            Must be greater than 0.
        **kwargs
            Keyword arguments to be passed to base MlsSim class constructor.
            See the MlsSim class documentation for details.
        
        Returns
        -------
        None.
    
        """
        self.ndim = 2
        self.nNodes = N*N
        self.nodes = np.indices((N, N), dtype='float64').reshape(2,-1).T / N
        super().__init__(N, **kwargs)
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.generateQuadraturePoints(kwargs['quadrature'])
        self.dudt = np.zeros(self.nNodes, dtype='float64')
        self.time = 0.0
        self.timestep = 0
        self.dt = dt
        ##### Augment nodes for periodic BCs #####
        self.periodicIndices = np.arange(0, self.nNodes)
        newInds1 = np.flatnonzero(self.nodes[:,0] < self.support)
        newInds2 = np.flatnonzero(self.nodes[:,0] > (1.0 - self.support))
        self.periodicIndices = np.hstack( (self.periodicIndices,
                                           newInds1,
                                           newInds2) )
        self.nodes = np.vstack( (self.nodes,
                                 self.nodes[newInds1] + [1.0, 0.0],
                                 self.nodes[newInds2] - [1.0, 0.0]) )
        newInds1 = np.flatnonzero(self.nodes[:,1] < self.support)
        newInds2 = np.flatnonzero(self.nodes[:,1] > (1.0 - self.support))
        self.periodicIndices = np.hstack( (self.periodicIndices, 
                                           self.periodicIndices[newInds1],
                                           self.periodicIndices[newInds2]) )
        self.nodes = np.vstack( (self.nodes,
                                 self.nodes[newInds1] + [0.0, 1.0],
                                 self.nodes[newInds2] - [0.0, 1.0]) )
        self.nodes, newInds = np.unique(self.nodes, return_index=True, axis=0)
        self.uIndices = np.flatnonzero(newInds<self.nNodes)
        self.periodicIndices = self.periodicIndices[newInds]
        self.setInitialConditions(u0)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}, {self.dt}, " \
               f"{self.u0.__name__}, {repr(self.velocity)}, " \
               f"{self.diffusivity}, Nquad={self.Nquad}, " \
               f"support={self.support*self.N}, form='{self.form}', " \
               f"quadrature='{self.quadrature}')"
    
    def setInitialConditions(self, u0):
        """Initialize the shape function coefficients for the given IC.

        Returns
        -------
        None.

        """
        self.u0 = u0
        self.uTime = 0.0
        try:
            if u0.shape == (self.nNodes,):
                self.u = u0
            else:
                raise Exception()
        except AttributeError: # u0 object has no attribute 'shape'
            self.u = u0(self.uNodes())
            if self.u.shape != (self.nNodes,):
                raise Exception()
        except:
            print(f"Error: u0 must be an array of shape ({self.nNodes},) or a "
                  f"function returning such an array and taking as input the "
                  f"array of (x,y) node coordinates with shape "
                  f"({self.nNodes}, 2).\nUsing default u = "
                  f"np.zeros({self.nNodes}, dtype='float64')")
            self.u = np.zeros(self.nNodes, dtype='float64')
        # pre-allocate arrays for constructing matrix equation for uI
        # this is the maximum possibly required size; not all will be used
        nMaxEntriesPerNode = int(self.nNodes*4*(self.support+0.25/self.N)**2)
        data = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='float64')
        indices = np.empty(self.nNodes * nMaxEntriesPerNode, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.uNodes()):
            inds = self.defineSupport(node)
            nEntries = len(inds)
            data[index:index+nEntries] = self.phi(node, self.nodes[inds])
            indices[index:index+nEntries] = self.periodicIndices[inds]
            indptr[iN] = index
            index += nEntries
        indptr[-1] = index
        A = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                           shape=(self.nNodes, self.nNodes) )
        self.uI = sp_la.spsolve(A, self.u)
    
    def computeSpatialDiscretization(self):
        """Assemble the system discretization matrices K, A, M in CSR format.
        K is the stiffness matrix from the diffusion term
        A is the advection matrix
        M is the mass matrix from the time derivative
        KA = K + A

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
            phi, gradphi = self.dphi(quad, self.nodes[indices])
            Kdata[index:index+nEntries] = np.ravel(
                gradphi @ (self.diffusivity @ gradphi.T) )
            Adata[index:index+nEntries] = np.ravel(
                np.outer(np.dot(gradphi, self.velocity), phi) )
            Mdata[index:index+nEntries] = np.ravel(np.outer(phi, phi))
            indices = self.periodicIndices[indices]
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
            # print(quad, indices)
            # if np.any(phi<0):
            #     print('Negative phi value detected!!!!!')
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
        self.A *= self.quadWeight
        self.M *= self.quadWeight
        self.KA = self.K + self.A
    
    def step(self, nSteps = 1, **kwargs):
        info = 0
        betas = np.array([0.25, 1.0/3.0, 0.5, 1.0], dtype='float64') ## RK4 ##
        # betas = np.array([1.0], dtype='float64') ## Forward Euler ##
        for i in range(nSteps):
            uTemp = self.uI
            for beta in betas:
                self.dudt, info = sp_la.cg(self.M, self.KA@uTemp, x0=self.dudt, **kwargs)
                # self.dudt = sp_la.spsolve(self.M, self.KA@self.uI)
                uTemp = self.uI + beta*self.dt*self.dudt
                if (info != 0):
                    print(f'solution failed with error code: {info}')
            self.uI = uTemp
            self.timestep += 1
        self.time = self.timestep * self.dt
    
    def uNodes(self):
        """Return the set of nodes on which the solution is computed.

        Returns
        -------
        nx2 numpy.ndarray, dtype='float64'
            Subset of self.nodes on which the solution is actually computed.

        """
        return self.nodes[self.uIndices]
    
    def solve(self):
        """Reconstruct the final solution vector, u, from shape functions.

        Returns
        -------
        None.

        """
        self.uTime = self.time
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.uNodes()):
            indices = self.defineSupport(node)
            self.u[iN] = self.uI[self.periodicIndices[indices]] @ \
                         self.phi(node, self.nodes[indices])
    
    def cond(self, order=2):
        """Computes the condition number of the mass matrix M.
        
        Parameters
        ----------
        order : {int, inf, -inf, ‘fro’}, optional
            Order of the norm. inf means numpy’s inf object. The default is 2.

        Returns
        -------
        float
            The condition number of the matrix.

        """
        return super().cond(self.M, None, order)
