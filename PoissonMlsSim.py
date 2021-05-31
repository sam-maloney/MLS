#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:20:15 2020

@author: Sam Maloney
"""

import mls
import pyamg
import scipy
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
    f : function object
        Function defining the forcing term throughout the domain.
    method : string, optional
        Name of method used for assembling the stiffness matrix.
    isBoundaryNode : numpy.ndarray, dtype='bool'
        Information on whether a given node is on the Dirichlet boundary.
    nBoundaryNodes : int
        Numer of nodes on the Dirichlet boundary.
    nInteriorNodes : int
        Numer of nodes NOT on the Dirichlet boundary.
    boundaryValues : numpy.ndarray, dtype='float64'
        Stored values of g() evaluated at the boundary nodes.
    
    Methods
    -------
    selectMethod(self, method, quadrature)
        Register the 'self.assembleStiffnesMatrix' method.
    solve(self, preconditioner=None, **kwargs):
        Solve for the approximate solution using an iterative solver.
        
    """
   
    def __init__(self, N, g, f=lambda x: 0., method='galerkin',
                 quadrature='gaussian', perturbation=0, seed=None, **kwargs):
        """Construct PoissonMlsSim by extending MlsSim constructor
    
        Parameters
        ----------
        N : integer
            Number of grid cells along one dimension. Must be greater than 0.
        g : function object
            Function defining the solution Dirichlet values along the boundary.
            The object must take an nx2 numpy.ndarray of points and return a
            1D numpy.ndarray of size n for the function values at those points.
        f : function object, optional
            Function defining the forcing term throughout the domain.
            The object must take an nx2 numpy.ndarray of points and return a
            1D numpy.ndarray of size n for the function values at those points.
        method : string, optional
            Method used for assembling the stiffness matrix.
            Must be either 'galerkin' or 'collocation'. Default is 'galerkin'.
        quadrature : string, optional
            Distribution of quadrature points in each cell.
            Must be one of 'gaussian', 'uniform' or 'vci'.
            Default is 'gaussian'.
        perturbation : float, optional
            Max amplitude of random perturbations added to node locations.
            Size is relative to grid spacing. Default is 0.
        seed : {None, int, array_like[ints], numpy.random.SeedSequence}, optional
            A seed to initialize the RNG. If None, then fresh, unpredictable
            entropy will be pulled from the OS. Default is None.
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
        self.nInteriorNodes = self.nNodes - self.nBoundaryNodes
        rng = np.random.Generator(np.random.PCG64(seed))
        self.nodes[~self.isBoundaryNode] += \
            rng.uniform(-self.dx*perturbation, self.dx*perturbation,
                        (self.nInteriorNodes, self.ndim))
        self.nodes[self.nodes < 0] = 0.
        self.nodes[self.nodes > 1] = 1.
        self.boundaryValues = g(self.nodes[self.isBoundaryNode]) \
                                .round(decimals=14)
        self.boundaryIndices = np.arange(self.nNodes)[self.isBoundaryNode]
        self.g = g
        self.f = f
        self.selectMethod(method, quadrature)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.N}, {self.g.__name__}, " \
               f"'{self.method}', '{self.quadrature}', Nquad={self.Nquad}, " \
               f"support={repr(self.support)}, form='{self.form}', " \
               f"basis='{self.basis.name}')"
    
    def selectMethod(self, method='galerkin', quadrature='gaussian'):
        """Register the 'self.assembleStiffnesMatrix' method.
        
        Parameters
        ----------
        method : string
            Method used for assembling the stiffness matrix.
            Must be either 'galerkin' or 'collocation'. Default is 'galerkin'.
        quadrature : string
            Distribution of quadrature points in each cell.
            Must be either 'uniform' or 'gaussian'. Default is 'gaussian'.

        Returns
        -------
        None.

        """
        self.method = method.lower()
        if self.method == 'galerkin':
            if quadrature.lower() == 'vci':
                self.assembleStiffnessMatrix = \
                    self.assembleGalerkinStiffnessMatrixLinearVCI
                    # self.assembleGalerkinStiffnessMatrixVCI
            else:
                self.assembleStiffnessMatrix = \
                    self.assembleGalerkinStiffnessMatrix
                self.generateQuadraturePoints(quadrature)
        elif self.method == 'collocation':
            self.assembleStiffnessMatrix = \
                self.assembleCollocationStiffnessMatrix
        else:
            print(f"Error: unkown assembly method '{method}'. "
                  f"Must be one of 'galerkin' or 'collocation'.")
    
    def assembleGalerkinStiffnessMatrixVCI(self):
        """Assemble the Galerkin system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        self.nQuads = self.Nquad**self.ndim * self.N**self.ndim
        nMaxEntries = int((self.nNodes * self.support.volume)**2 * self.nQuads)
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        # compute Gauss-Legendre quadrature points
        offsets, weights = scipy.special.roots_legendre(self.Nquad)
        offsets /= (2*self.N)
        weights /= (2*self.N)
        quads = np.zeros((1, self.ndim))
        quadWeights = np.array([1.])
        LR = np.vstack((np.repeat(1/(2*self.N), len(offsets)), offsets)).T
        UD = np.vstack((offsets, np.repeat(1/(2*self.N), len(offsets)))).T
        for i in range(self.ndim):
            quads = np.concatenate( [quads + offset*np.eye(self.ndim)[i]
                                     for offset in offsets] )
            quadWeights = np.concatenate( [quadWeights * weight
                                           for weight in weights] )
        # build matrix for interior nodes
        index = 0
        self.b = np.zeros(self.nNodes)
        A = np.zeros((self.nNodes, 3, 3))
        r = np.empty((self.nNodes, self.ndim, 3))
        xi = np.zeros((self.nNodes, self.ndim, 3))
        for iC, cell in enumerate(( np.indices(np.repeat(self.N, self.ndim),
                dtype='float64').T.reshape(-1, self.ndim) + 0.5 ) / self.N):
            r.fill(0)
            store = []
            if A.any() or r.any() or xi.any():
                raise SystemExit('Error: temporary arrays not zeroed properly!!!')
            for iQ, quad in enumerate(cell + quads): # interior
                indices, phis, gradphis = self.dphi(quad)
                disps = quad - self.nodes[indices]
                store.append((indices, phis, gradphis, disps))
                P = np.hstack((np.ones((len(indices), 1)), disps))
                A[indices] += quadWeights[iQ] * \
                    np.apply_along_axis(lambda x: np.outer(x,x), 1, P)
                r[indices,:,0] -= gradphis * quadWeights[iQ]
                r[indices,0,1] -= phis * quadWeights[iQ]
                r[indices,1,2] -= phis * quadWeights[iQ]
                r[indices,:,1:3] -= quadWeights[iQ] * np.apply_along_axis(
                    lambda x: np.outer(x[0:2], x[2:4]), 1,
                    np.hstack((gradphis, disps)))
                self.b[indices] += self.f(quad) * phis * quadWeights[iQ]
            for iQ, quad in enumerate(cell - LR): # left
                indices, phis = self.phi(quad)
                r[indices,0,0] -= phis * weights[iQ]
                r[indices,0,1:3] -= phis.reshape(-1,1) * weights[iQ] * \
                    (quad - self.nodes[indices])
            for iQ, quad in enumerate(cell + LR): # right
                indices, phis = self.phi(quad)
                r[indices,0,0] += phis * weights[iQ]
                r[indices,0,1:3] += phis.reshape(-1,1) * weights[iQ] * \
                    (quad - self.nodes[indices])
            for iQ, quad in enumerate(cell - UD): # down
                indices, phis = self.phi(quad)
                r[indices,1,0] -= phis * weights[iQ]
                r[indices,1,1:3] -= phis.reshape(-1,1) * weights[iQ] * \
                    (quad - self.nodes[indices])
            for iQ, quad in enumerate(cell + UD): # up
                indices, phis = self.phi(quad)
                r[indices,1,0] += phis * weights[iQ]
                r[indices,1,1:3] += phis.reshape(-1,1) * weights[iQ] * \
                    (quad - self.nodes[indices])
            cell_indices = np.unique(np.concatenate([x[0] for x in store]))
            for i in cell_indices:
                lu, piv = la.lu_factor(A[i], True, False)
                for j in range(self.ndim):
                    xi[i,j] = la.lu_solve((lu, piv), r[i,j], 0, True, False)
            for iQ, (indices, phis, gradphis, disps) in enumerate(store):
                nEntries = len(indices)**2
                data[index:index+nEntries] = quadWeights[iQ] * \
                    np.ravel((gradphis + xi[indices,:,0] +
                              xi[indices,:,1] * disps[:,0:1] +
                              xi[indices,:,2] * disps[:,1:2]) @ gradphis.T)
                row_ind[index:index+nEntries] = np.repeat(indices,len(indices))
                col_ind[index:index+nEntries] = np.tile(indices, len(indices))
                index += nEntries
            A[cell_indices] = 0
            xi[cell_indices] = 0
        # assemble the triplets into the sparse stiffness matrix
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                                shape=(self.nNodes, self.nNodes) )
        ##### Apply BCs using Lagrange multipliers #####
        nMaxEntries = int( (self.nNodes * self.support.volume)**2
                          * self.nBoundaryNodes )
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
            indices, phis = self.phi(node)
            nEntries = len(indices)
            data[index:index+nEntries] = phis
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                            shape=(self.nNodes, self.nBoundaryNodes) )
        # G *= -1.0
        self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
        self.b = np.concatenate(( self.b, self.boundaryValues ))
    
    # def assembleGalerkinStiffnessMatrixLinearVCI(self):
    #     """Assemble the Galerkin system stiffness matrix K in CSR format.

    #     Returns
    #     -------
    #     None.

    #     """
    #     # pre-allocate arrays for stiffness matrix triplets
    #     # these are the maximum possibly required sizes; not all will be used
    #     self.nQuads = self.Nquad**self.ndim * self.N**self.ndim
    #     nMaxEntries = int((self.nNodes * self.support.volume)**2 * self.nQuads)
    #     data = np.zeros(nMaxEntries)
    #     row_ind = np.zeros(nMaxEntries, dtype='uint32')
    #     col_ind = np.zeros(nMaxEntries, dtype='uint32')
    #     # compute Gauss-Legendre quadrature points
    #     offsets, weights = scipy.special.roots_legendre(self.Nquad)
    #     offsets /= (2*self.N)
    #     weights /= (2*self.N)
    #     quads = np.zeros((1, self.ndim))
    #     quadWeights = np.array([1.])
    #     LR = np.vstack((np.repeat(1/(2*self.N), len(offsets)), offsets)).T
    #     UD = np.vstack((offsets, np.repeat(1/(2*self.N), len(offsets)))).T
    #     for i in range(self.ndim):
    #         quads = np.concatenate( [quads + offset*np.eye(self.ndim)[i]
    #                                  for offset in offsets] )
    #         quadWeights = np.concatenate( [quadWeights * weight
    #                                        for weight in weights] )
    #     # build matrix for interior nodes
    #     index = 0
    #     self.b = np.zeros(self.nNodes)
    #     gradphiSums = np.empty((self.nNodes, self.ndim))
    #     areas = np.empty(self.nNodes)
    #     xis = np.empty((self.nNodes, self.ndim))
    #     for iC, cell in enumerate(( np.indices(np.repeat(self.N, self.ndim),
    #             dtype='float64').T.reshape(-1, self.ndim) + 0.5 ) / self.N):
    #         # phiSums = np.zeros(self.nNodes)
    #         gradphiSums.fill(0)
    #         areas.fill(0)
    #         xis.fill(0)
    #         store = []
    #         for iQ, quad in enumerate(cell + quads):
    #             indices, phis, gradphis = self.dphi(quad)
    #             store.append((indices, phis, gradphis))
    #             areas[indices] += quadWeights[iQ]
    #             # phiSums[indices] += phis * quadWeights[iQ]
    #             gradphiSums[indices] += gradphis * quadWeights[iQ]
    #             self.b[indices] += self.f(quad) * phis * quadWeights[iQ]
    #         nonZeroInds = np.flatnonzero(areas)
    #         xis[nonZeroInds] = -gradphiSums[nonZeroInds] / \
    #                             areas[nonZeroInds].reshape(-1,1)
    #         # for iQ, quad in enumerate(cell - LR): # left
    #         #     if quad[0] < 1e-3: continue
    #         #     indices, phis = self.phi(quad)
    #         #     boolInds = np.isin(indices, nonZeroInds)
    #         #     indices = indices[boolInds]
    #         #     xis[indices, 0] -= phis[boolInds] * weights[iQ] / areas[indices]
    #         # for iQ, quad in enumerate(cell + LR): # right
    #         #     if quad[0] > 0.999: continue
    #         #     indices, phis = self.phi(quad)
    #         #     boolInds = np.isin(indices, nonZeroInds)
    #         #     indices = indices[boolInds]
    #         #     xis[indices, 0] += phis[boolInds] * weights[iQ] / areas[indices]
    #         # for iQ, quad in enumerate(cell - UD): # down
    #         #     if quad[1] < 1e-3: continue
    #         #     indices, phis = self.phi(quad)
    #         #     boolInds = np.isin(indices, nonZeroInds)
    #         #     indices = indices[boolInds]
    #         #     xis[indices, 1] -= phis[boolInds] * weights[iQ] / areas[indices]
    #         # for iQ, quad in enumerate(cell + UD): # up
    #         #     if quad[1] > 0.999: continue
    #         #     indices, phis = self.phi(quad)
    #         #     boolInds = np.isin(indices, nonZeroInds)
    #         #     indices = indices[boolInds]
    #         #     xis[indices, 1] += phis[boolInds] * weights[iQ] / areas[indices]
            
    #         # nonInteriorX = (self.nodes[:,0] < self.support.size) \
    #         #              + (self.nodes[:,0] > (1 - self.support.size))
    #         # nonInteriorY = (self.nodes[:,1] < self.support.size) \
    #         #              + (self.nodes[:,1] > (1 - self.support.size))
    #         # xis[nonInteriorX,0] = 0.
    #         # xis[nonInteriorY,1] = 0.
            
    #         for iQ, (indices, phis, gradphis) in enumerate(store):
    #             nEntries = len(indices)**2
    #             data[index:index+nEntries] = quadWeights[iQ] * \
    #                 np.ravel((gradphis + xis[indices]) @ gradphis.T)
    #             row_ind[index:index+nEntries] = np.repeat(indices,len(indices))
    #             col_ind[index:index+nEntries] = np.tile(indices, len(indices))
    #             index += nEntries
    #     # assemble the triplets into the sparse stiffness matrix
    #     inds = np.flatnonzero(data.round(decimals=14,out=data))
    #     self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
    #                             shape=(self.nNodes, self.nNodes) )
    #     ##### Apply BCs using Lagrange multipliers #####
    #     nMaxEntries = int( (self.nNodes * self.support.volume)**2
    #                       * self.nBoundaryNodes )
    #     data = np.zeros(nMaxEntries)
    #     row_ind = np.zeros(nMaxEntries, dtype='uint32')
    #     col_ind = np.zeros(nMaxEntries, dtype='uint32')
    #     index = 0
    #     for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
    #         indices, phis = self.phi(node)
    #         nEntries = len(indices)
    #         data[index:index+nEntries] = phis
    #         row_ind[index:index+nEntries] = indices
    #         col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
    #         index += nEntries
    #     inds = np.flatnonzero(data.round(decimals=14,out=data))
    #     G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
    #                         shape=(self.nNodes, self.nBoundaryNodes) )
    #     # G *= -1.0
    #     self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
    #     self.b = np.concatenate(( self.b, self.boundaryValues ))
    
    def assembleGalerkinStiffnessMatrixLinearVCI(self):
        """Assemble the Galerkin system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        self.nQuads = self.Nquad**self.ndim * self.N**self.ndim
        nMaxEntries = int((self.nNodes * self.support.volume)**2 * self.nQuads)
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        # compute Gauss-Legendre quadrature points
        offsets, weights = scipy.special.roots_legendre(self.Nquad)
        offsets /= (2*self.N)
        weights /= (2*self.N)
        # quads = np.zeros((1, self.ndim))
        # quadWeights = np.array([1.])
        
        quads = ( np.indices(np.repeat(self.N, self.ndim),
                dtype='float64').T.reshape(-1, self.ndim) + 0.5 ) / self.N
        quadWeights = np.repeat(1., len(quads))
        
        # LR = np.vstack((np.repeat(1/(2*self.N), len(offsets)), offsets)).T
        # UD = np.vstack((offsets, np.repeat(1/(2*self.N), len(offsets)))).T
        for i in range(self.ndim):
            quads = np.concatenate( [quads + offset*np.eye(self.ndim)[i]
                                     for offset in offsets] )
            quadWeights = np.concatenate( [quadWeights * weight
                                           for weight in weights] )
        # build matrix for interior nodes
        index = 0
        self.b = np.zeros(self.nNodes)
        gradphiSums = np.zeros((self.nNodes, self.ndim))
        areas = np.zeros(self.nNodes)
        xis = np.zeros((self.nNodes, self.ndim))
        store = []
        # for iC, cell in enumerate(( np.indices(np.repeat(self.N, self.ndim),
        #         dtype='float64').T.reshape(-1, self.ndim) + 0.5 ) / self.N):
        for iQ, quad in enumerate(quads):
            indices, phis, gradphis = self.dphi(quad)
            store.append((indices, gradphis, quadWeights[iQ]))
            areas[indices] += quadWeights[iQ]
            gradphiSums[indices] += gradphis * quadWeights[iQ]
            self.b[indices] += self.f(quad) * phis * quadWeights[iQ]
        
        xis = -gradphiSums / areas.reshape(-1,1)
            
        for iQ, (indices, gradphis, quadWeight) in enumerate(store):
            nEntries = len(indices)**2
            data[index:index+nEntries] = quadWeight * \
                np.ravel((gradphis + xis[indices]) @ gradphis.T)
            row_ind[index:index+nEntries] = np.repeat(indices,len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
        # assemble the triplets into the sparse stiffness matrix
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                                shape=(self.nNodes, self.nNodes) )
        ##### Apply BCs using Lagrange multipliers #####
        nMaxEntries = int( (self.nNodes * self.support.volume)**2
                          * self.nBoundaryNodes )
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
            indices, phis = self.phi(node)
            nEntries = len(indices)
            data[index:index+nEntries] = phis
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                            shape=(self.nNodes, self.nBoundaryNodes) )
        # G *= -1.0
        self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
        self.b = np.concatenate(( self.b, self.boundaryValues ))
    
    def assembleGalerkinStiffnessMatrix(self):
        """Assemble the Galerkin system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # #### This code applies BCs directly; more ill-conditioned #####
        # # pre-allocate arrays for stiffness matrix triplets
        # # these are the maximum possibly required sizes; not all will be used
        # nMaxEntries = int((self.nNodes * self.support.volume)**2 * self.nQuads)
        # data = np.zeros(nMaxEntries, dtype='float64')
        # row_ind = np.zeros(nMaxEntries, dtype='uint32')
        # col_ind = np.zeros(nMaxEntries, dtype='uint32')
        # # build matrix for interior nodes
        # index = 0
        # for iQ, quad in enumerate(self.quads):
        #     indices, gradphis = self.dphi(quad)[0:3:2]
        #     i_inds = indices[~self.isBoundaryNode[indices]] # interior nodes
        #     nEntries = len(i_inds)*len(indices)
        #     data[index:index+nEntries] = \
        #         np.ravel(gradphis[~self.isBoundaryNode[indices]] @ gradphis.T)
        #     row_ind[index:index+nEntries] = np.repeat(i_inds, len(indices))
        #     col_ind[index:index+nEntries] = np.tile(indices, len(i_inds))
        #     index += nEntries
        # for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
        #     indices, phis = self.phi(node)
        #     nEntries = len(indices)
        #     data[index:index+nEntries] = phis
        #     row_ind[index:index+nEntries] = np.repeat(self.boundaryIndices[iN], nEntries)
        #     col_ind[index:index+nEntries] = indices
        #     index += nEntries
        # inds = np.flatnonzero(data.round(decimals=14,out=data))
        # # assemble the triplets into the sparse stiffness matrix
        # self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
        #                         shape=(self.nNodes, self.nNodes) )
        # self.b = self.f(self.nodes)
        # self.b[self.isBoundaryNode] = self.boundaryValues
        
        # pre-allocate arrays for stiffness matrix triplets
        # these are the maximum possibly required sizes; not all will be used
        nMaxEntries = int((self.nNodes * self.support.volume)**2 * self.nQuads)
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        self.b = np.zeros(self.nNodes)
        # build matrix for interior nodes
        index = 0
        for iQ, quad in enumerate(self.quads):
            indices, phis, gradphis = self.dphi(quad)
            nEntries = len(indices)**2
            data[index:index+nEntries] = np.ravel(gradphis @ gradphis.T) *\
                self.quadWeights[iQ]
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
            self.b[indices] += self.f(quad) * phis * self.quadWeights[iQ]
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        # assemble the triplets into the sparse stiffness matrix
        self.K = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                                shape=(self.nNodes, self.nNodes) )
        ##### Apply BCs using Lagrange multipliers #####
        nMaxEntries = int( (self.nNodes * self.support.volume)**2
                          * self.nBoundaryNodes )
        data = np.zeros(nMaxEntries)
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.nodes[self.isBoundaryNode]):
            indices, phis = self.phi(node)
            nEntries = len(indices)
            data[index:index+nEntries] = phis
            row_ind[index:index+nEntries] = indices
            col_ind[index:index+nEntries] = np.repeat(iN, nEntries)
            index += nEntries
        inds = np.flatnonzero(data.round(decimals=14,out=data))
        G = sp.csr_matrix( (data[inds], (row_ind[inds], col_ind[inds])),
                            shape=(self.nNodes, self.nBoundaryNodes) )
        # G *= -1.0
        self.K = sp.bmat([[self.K, G], [G.T, None]], format='csr')
        self.b = np.concatenate(( self.b, self.boundaryValues ))
    
    def assembleCollocationStiffnessMatrix(self):
        """Assemble the collocation system stiffness matrix K in CSR format.

        Returns
        -------
        None.

        """
        # pre-allocate arrays for constructing stiffness matrix
        # this is the maximum possibly required size; not all will be used
        nMaxEntries = int(self.nNodes**2 * self.support.volume)
        data = np.empty(nMaxEntries)
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
                data[index:index+nEntries] = -d2phis.sum(axis=1)
            indices[index:index+nEntries] = inds
            index += nEntries
        # print(f"{index}/{nMaxEntries} used for spatial discretization")
        indptr[-1] = index
        self.K = sp.csr_matrix( (data[0:index], indices[0:index], indptr),
                                shape=(self.nNodes, self.nNodes) )
        self.b = self.f(self.nodes)
        self.b[self.isBoundaryNode] = self.boundaryValues
    
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
        elif preconditioner.lower() == 'amg':
            ml = pyamg.ruge_stuben_solver(self.K)
            self.M = ml.aspreconditioner()
        elif preconditioner.lower() == 'ilu':
            ilu = sp_la.spilu(self.K)
            Mx = lambda x: ilu.solve(x)
            self.M = sp_la.LinearOperator(self.K.shape, Mx)
        elif preconditioner.lower() == 'jacobi':
            if self.method.lower() == 'collocation':
                self.M = sp_la.inv( sp.diags(self.K.diagonal(), format='csc') )
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
        self.u = np.empty(self.nNodes)
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
