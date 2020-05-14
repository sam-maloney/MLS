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
    dt : float
        Time interval between each successive timestep.
    timestep : int
        Current timestep of the simulation.
    time : float
        Current time of the simulation; equal to timestep*dt.
    dudt : numpy.ndarray, dtype='float64'
        (self.nNodes,) array of most recently computed time derivative.
    periodicIndices : numpy.ndarray, dtype='int'
        Inidices mapping periodic boundary nodes to their real solution nodes.
    uIndices : numpy.ndarray, dtype='int'
        Indices of the real solution nodes in the full periodic nodes array.
    
    Methods
    -------
    setInitialConditions(self, u0):
        Initialize the shape function coefficients for the given IC.
    computeSpatialDiscretization(self):
        Assemble the system discretization matrices K, A, M in CSR format.
    precondition(self, preconditioner=None, P=None):
        Generate and/or store the preconditioning matrix P.
    step(self, nSteps = 1, **kwargs):
        Advance the simulation a given number of timesteps.
    uNodes(self):
        Return the set of solution nodes that excludes periodic repeats.
    solve(self):
        Reconstruct the final solution vector, u, from shape functions.
    cond(self, order=2):
        Compute the condition number of the (preconditioned) mass matrix M.
    """
    
    def __init__(self, N, dt, u0, velocity, diffusivity,
                 quadrature='gaussian', perturbation=0, seed=None, **kwargs):
        """Initialize attributes of ConvectionDiffusion simulation class
        Extends MlsSim.__init__() constructor
    
        Parameters
        ----------
        N : int
            Number of grid cells along one dimension. Must be greater than 0.
        dt : float
            Time interval between each successive timestep.
        u0 : {numpy.ndarray, function object}
            Initial conditions for the simulation.
            Must be an array of shape (self.nNodes,) or a function returning
            such an array and taking as input the array of (x,y) node
            coordinates with shape ({self.nNodes}, 2).
        velocity : np.array([vx,vy,vz], dtype='float64')
            Background velocity of the fluid.
        diffusivity : {numpy.ndarray, float}
            Diffusion coefficient for the quantity of interest.
            If an array, it must have shape (ndim,ndim). If a float, it will
            be converted to diffusivity*np.eye(ndim, dtype='float64').
        quadrature : string, optional
            Distribution of quadrature points in each cell.
            Must be either 'gaussian' or 'uniform'. Default is 'gaussian'.
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
        super().__init__(N, **kwargs)
        self.nNodes = N**self.ndim
        self.nodes = np.indices(np.repeat(N, self.ndim), dtype='float64') \
                    .reshape(self.ndim,-1).T / N
        rng = np.random.Generator(np.random.PCG64(seed))
        self.nodes += rng.uniform(-self.dx*perturbation, self.dx*perturbation,
                                  self.nodes.shape)
        self.velocity = velocity
        if isinstance(diffusivity, np.ndarray):
            self.diffusivity = diffusivity
        else:
            self.diffusivity = np.array(diffusivity, dtype='float64')
            if self.diffusivity.shape != (self.ndim,self.ndim):
                self.diffusivity = diffusivity * np.eye(self.ndim, dtype='float64')
        if self.diffusivity.shape != (self.ndim,self.ndim):
            raise SystemExit(f"diffusivity must be (or be convertible to) a "
                f"numpy.ndarray with shape ({self.ndim},{self.ndim}) for "
                f"ndim = {self.ndim}.")
        self.generateQuadraturePoints(quadrature)
        self.dudt = np.zeros(self.nNodes, dtype='float64')
        self.time = 0.0
        self.timestep = 0
        self.dt = dt
        ##### Augment nodes for periodic BCs #####
        self.periodicIndices = np.arange(0, self.nNodes)
        for i in range(self.ndim):
            newInds1 = np.flatnonzero(self.nodes[:,i] < self.support.size)
            newInds2 = np.flatnonzero(self.nodes[:,i] > (1.0 - self.support.size))
            self.periodicIndices = np.hstack((self.periodicIndices, 
                                              self.periodicIndices[newInds1],
                                              self.periodicIndices[newInds2]))
            self.nodes = np.vstack((self.nodes,
                                    self.nodes[newInds1] + np.eye(self.ndim)[i],
                                    self.nodes[newInds2] - np.eye(self.ndim)[i]))
        self.nodes, newInds = np.unique(self.nodes, return_index=True, axis=0)
        self.uIndices = np.flatnonzero(newInds < self.nNodes)
        self.periodicIndices = self.periodicIndices[newInds]
        self.setInitialConditions(u0)
    
    def __repr__(self):
        diffusivity_repr = ' '.join(repr(self.diffusivity).split())
        return f"{self.__class__.__name__}({self.N}, {self.dt}, " \
               f"{self.u0.__name__}, {repr(self.velocity)}, " \
               f"{diffusivity_repr}, '{self.quadrature}', ndim={self.ndim}, " \
               f"Nquad={self.Nquad}, support={repr(self.support)}, " \
               f"form='{self.form}', basis='{self.basis.name}')"
    
    def setInitialConditions(self, u0):
        """Initialize the shape function coefficients for the given IC.

        Returns
        -------
        None.

        """
        self.u0 = u0
        self.uTime = 0.0
        if isinstance(u0, np.ndarray) and u0.shape == (self.nNodes,):
            self.u = u0
        else:
            try:
                self.u = u0(self.uNodes())
                if self.u.shape != (self.nNodes,):
                    raise Exception()
            except:
                raise SystemExit(f"u0 must be an array of shape "
                    f"({self.nNodes},) or a function returning such an array "
                    f"and taking as input the array of (x,y) node coordinates "
                    f"with shape ({self.nNodes}, 2).")
        # pre-allocate arrays for constructing matrix equation for uI
        # this is the maximum possibly required size; not all will be used
        nMaxEntries = int( self.support.volume * (self.N+1)**self.ndim
                          *self.nNodes )
        data = np.empty(nMaxEntries, dtype='float64')
        indices = np.empty(nMaxEntries, dtype='uint32')
        indptr = np.empty(self.nNodes+1, dtype='uint32')
        index = 0
        for iN, node in enumerate(self.uNodes()):
            inds, phis = self.phi(node)
            nEntries = len(inds)
            data[index:index+nEntries] = phis
            indices[index:index+nEntries] = self.periodicIndices[inds]
            indptr[iN] = index
            index += nEntries
        indptr[-1] = index
        # print(f"{index}/{nMaxEntries} used for u0")
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
        nMaxEntries = int( (self.support.volume * (self.N+1)**self.ndim)**2
                           *self.nQuads )
        Kdata = np.zeros(nMaxEntries, dtype='float64')
        Adata = np.zeros(nMaxEntries, dtype='float64')
        Mdata = np.zeros(nMaxEntries, dtype='float64')
        row_ind = np.zeros(nMaxEntries, dtype='uint32')
        col_ind = np.zeros(nMaxEntries, dtype='uint32')
        # build matrix for interior nodes
        index = 0
        for iQ, quad in enumerate(self.quads):
            indices, phis, gradphis = self.dphi(quad)
            nEntries = len(indices)**2
            Kdata[index:index+nEntries] = self.quadWeights[iQ] * \
                np.ravel( gradphis @ (self.diffusivity @ gradphis.T) )
            Adata[index:index+nEntries] = self.quadWeights[iQ] * \
                np.ravel( np.outer(np.dot(gradphis, self.velocity), phis) )
            Mdata[index:index+nEntries] = self.quadWeights[iQ] * \
                np.ravel( np.outer(phis, phis) )
            indices = self.periodicIndices[indices]
            row_ind[index:index+nEntries] = np.repeat(indices, len(indices))
            col_ind[index:index+nEntries] = np.tile(indices, len(indices))
            index += nEntries
            # print(quad, indices)
            # if np.any(phis<0):
            #     print('Negative phi value detected!!!!!')
        # print(f"{index}/{nMaxEntries} used for spatial discretization")
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
        # self.KA = self.K + self.A
        self.KA = self.A - self.K
        self.P = None
    
    def precondition(self, preconditioner=None, P=None):
        """Generate and/or store the preconditioning matrix P.

        Parameters
        ----------
        preconditioner : {string, None}, optional
            Which preconditioning method to use. If P is not given, then it
            must be one of 'jacobi', 'ilu', or None. If P is given, then any
            string can be given as the name for P; if None is given, it will
            default to 'user_defined'. The default is None.
        P : {scipy.sparse.linalg.LinearOperator, None}, optional
            Used to directly specifiy the linear operator to be used.
            The default is None.

        Returns
        -------
        None.

        """
        self.preconditioner = preconditioner
        if P != None:
            self.P = P
            if self.preconditioner == None:
                self.preconditioner = 'user_defined'
            return
        if self.preconditioner == None:
            self.P = None
        elif preconditioner.lower() == 'ilu':
            ilu = sp_la.spilu(self.M)
            self.P = sp_la.LinearOperator(self.M.shape, lambda x: ilu.solve(x))
        elif preconditioner.lower() == 'jacobi':
            self.P = sp_la.inv( sp.diags( self.M.diagonal(), format='csc',
                                          dtype='float64' ) )
        else:
            print(f"Error: unkown preconditioner '{preconditioner}'. "
                  f"Must be one of 'ilu' or 'jacobi' (or None). "
                  f"Defaulting to None.")
    
    def step(self, nSteps = 1, **kwargs):
        """Advance the simulation a given number of timesteps.

        Parameters
        ----------
        nSteps : int, optional
            Number of timesteps to compute. The default is 1.
        **kwargs
            Used to specify optional arguments passed to the linear solver.
            Note that kwargs["M"] will be overwritten, use self.precon(...)
            instead to generate or specify a preconditioner.

        Returns
        -------
        None.

        """
        kwargs["M"] = self.P
        info = 0
        betas = np.array([0.25, 1.0/3.0, 0.5, 1.0], dtype='float64') ## RK4 ##
        # betas = np.array([1.0], dtype='float64') ## Forward Euler ##
        for i in range(nSteps):
            uTemp = self.uI
            for beta in betas:
                self.dudt, info = sp_la.cg(self.M, self.KA@uTemp,
                                           x0=self.dudt, **kwargs)
                # self.dudt = sp_la.spsolve(self.M, self.KA@uTemp)
                uTemp = self.uI + beta*self.dt*self.dudt
                if (info != 0):
                    print(f'solution failed with error code: {info}')
            self.uI = uTemp
            self.timestep += 1
        self.time = self.timestep * self.dt
    
    def uNodes(self):
        """Return the set of solution nodes that excludes periodic repeats.
        Overrides the superclass MlsSim.uNodes() method.

        Returns
        -------
        nx2 numpy.ndarray, dtype='float64'
            Subset of self.nodes on which the solution is actually computed.

        """
        return self.nodes[self.uIndices]
    
    def solve(self):
        """Reconstruct the final solution vector, u, from shape functions.
        Implements the superclass MlsSim.solve() abstract method.

        Returns
        -------
        None.

        """
        self.uTime = self.time
        self.u = np.empty(self.nNodes, dtype='float64')
        for iN, node in enumerate(self.uNodes()):
            indices, phis = self.phi(node)
            self.u[iN] = self.uI[self.periodicIndices[indices]] @ phis
    
    def cond(self, order=2):
        """Compute the condition number of the (preconditioned) mass matrix M.
        Utilizes the superclass MlsSim.cond() method.
        
        Parameters
        ----------
        order : {int, inf, -inf, ‘fro’}, optional
            Order of the norm. inf means numpy’s inf object. The default is 2.

        Returns
        -------
        float
            The condition number of the matrix.

        """
        return super().cond(self.M, self.P, order)
