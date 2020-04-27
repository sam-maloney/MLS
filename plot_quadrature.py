#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation using moving least squares (MLS)

@author: Sam Maloney
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# from PoissonMlsSim import PoissonMlsSim
# from timeit import default_timer

# # wavenumber for boundary function u(x,1) = g(x,y) = sinh(k*pi)
# k = 1
# def g(points):
#     k = 1
#     return np.sin(k*np.pi*points[:,0]) * np.sinh(k*np.pi*points[:,1])
            
# # mls = MlsSim(10)
# # mls.assembleStiffnessMatrix()

# N=64

# kwargs={
#     'Nquad' : 2,
#     'form' : 'quartic',
#     'method' : 'galerkin',
#     'quadrature' : 'gaussian',
#     'basis' : 'linear'}

# tolerance = 1e-10
# precon = 'ilu'

# # allocate arrays for convergence testing
# start = 1.1
# stop = 2.5
# step = 0.1
# nSamples = round((stop-start)/step) + 1
# supports = np.linspace(start, stop, num=nSamples)
# times = np.empty(nSamples, dtype='float64')
# conds = np.empty(nSamples, dtype='float64')
# fails = np.empty(nSamples, dtype='bool')
# errors = np.empty(nSamples, dtype='float64')

# # loop over N to test convergence where N is the number of
# # grid cells along one dimension, each cell forms 2 triangles
# # therefore number of nodes equals (N+1)*(N+1)
# for iS, support in enumerate(supports):
    
#     print('support =', support)
#     kwargs['support'] = support
    
#     start_time = default_timer()
    
#     # allocate arrays and compute boundary values
#     mlsSim = PoissonMlsSim(N, g, **kwargs)
    
#     # Assemble the stiffness matrix and solve for the approximate solution
#     mlsSim.assembleStiffnessMatrix()
#     mlsSim.solve(tol=tolerance, atol=tolerance, preconditioner=precon)
    
#     end_time = default_timer()
    
#     # compute the analytic solution and error norms
#     u_exact = g(mlsSim.nodes)
#     # E_inf = np.linalg.norm(mlsSim.u - u_exact, np.inf)
#     errors[iS] = np.linalg.norm(mlsSim.u - u_exact)/N
    
#     times[iS] = end_time-start_time
#     conds[iS] = mlsSim.cond('fro')
#     fails[iS] = (mlsSim.info != 0)
    
#     print('Condition Number =', conds[iS])
#     print(f'Elapsed time = {end_time-start_time} s\n')
    
# ##### End of loop over supports #####

supports = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3,
        2.4, 2.5])

errors_1 = np.array([7.66018606e-04, 6.95673162e-04, 5.94471774e-04, 5.19077244e-04,
       5.15951112e-04, 5.73608493e-04, 3.97701157e-04, 2.21282998e-04,
       4.13272002e-04, 7.50717383e-04, 1.25765124e-03, 1.94317927e-03,
       2.78504148e-03, 3.89518717e-03, 5.56638975e-03])
times_1 = np.array([1.7624402, 1.7151323, 1.7292107, 1.7676456, 1.7691637, 1.9633156,
       1.9146628, 1.9548862, 1.9371683, 1.9137728, 1.948194 , 2.1338077,
       2.1153125, 2.1930577, 2.1380175])
conds_1 = np.array([  4481.0030009 ,   4481.00304233,   4481.00308968,   4481.00254355,
         4481.00288892,   4481.02895344,   4481.06044654,   4481.05532094,
         4481.07190518,   4481.34492314,   4486.12232345, 401965.05646398,
         5232.39978688,   5059.99527773,   5439.43989971])
fails_1 = np.array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False])
    
errors_2_u = np.array([1.49785191e-04, 3.89414720e-04, 3.87896198e-04, 2.69326225e-04,
       1.05687630e-04, 1.30866723e-04, 1.20673472e-04, 5.30788320e-05,
       6.75542606e-05, 1.45536791e-04, 1.61696081e-04, 1.46348782e-04,
       3.16131748e-04, 6.10739465e-04, 8.86079641e-04])
times_2_u = np.array([4.9899054, 4.8919503, 4.9130391, 4.9610848, 4.9626615, 5.0047084,
       5.0878707, 5.1876859, 5.2655669, 5.4312303, 5.5067112, 5.5512028,
       5.9398646, 6.1552837, 6.4015869])
conds_2_u = np.array([4481.01251318, 4481.00778236, 4481.04335457, 4481.05448897,
       4481.0451009 , 4481.02995549, 4481.01958291, 4481.02337537,
       4481.02098575, 4481.02937899, 4481.11028112, 4482.54274995,
       4482.47995639, 4481.41103036, 4481.24320783])
fails_2_u = np.array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False])

errors_2_g = np.array([4.07025278e-04, 8.32412671e-05, 2.17976735e-04, 1.19917414e-04,
       1.34526285e-04, 2.18087597e-04, 1.65011850e-04, 7.16992916e-05,
       1.12611105e-04, 2.29297709e-04, 3.22932158e-04, 3.19372777e-04,
       2.35196792e-04, 1.24333250e-04, 1.55512326e-04])
times_2_g = np.array([4.9351403, 4.9485791, 5.0074687, 4.9649565, 5.1035687, 5.0069864,
       5.0991624, 5.2239478, 5.2989786, 5.5495202, 5.5313193, 5.6405585,
       6.0028399, 6.1805051, 6.2307714])
conds_2_g = np.array([4481.0134159 , 4481.01116075, 4481.0510877 , 4481.0512882 ,
       4481.04032282, 4481.02520923, 4481.01707296, 4481.01306971,
       4481.01729615, 4481.02954445, 4481.07387741, 4482.83741428,
       4482.60921903, 4481.32917592, 4481.20656739])
fails_2_g = np.array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False])

errors_3_u = np.array([4.03950555e-05, 2.01749270e-04, 2.52811667e-04, 1.73751013e-04,
       1.11752659e-04, 6.30478298e-05, 3.62655136e-05, 4.92525707e-05,
       6.74381223e-05, 4.27307929e-05, 7.44486298e-05, 9.68517272e-05,
       1.71654954e-04, 3.05826033e-04, 3.94117889e-04])
times_3_u = np.array([ 9.9247762, 10.1260761,  9.9763662, 10.2358302, 10.1894392,
       10.3808283, 10.5181621, 10.6462053, 10.9618434, 11.1979158,
       11.4307049, 12.5192172, 12.3285874, 12.6756068, 13.0170278])
conds_3_u = np.array([4481.01118692, 4481.01979124, 4481.05320783, 4481.04827662,
       4481.04278871, 4481.04463618, 4481.03400582, 4481.02602347,
       4481.02133919, 4481.0293601 , 4481.12674135, 4483.30990965,
       4481.81148479, 4481.29197624, 4481.2509704 ])
fails_3_u = np.array([False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False])
    
    
# ##### Begin Plotting Routines #####

# # clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,2.5)
plt.subplots_adjust(wspace = 0.35)

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot the minimum E_2 error norms
plt.subplot(131)
plt.semilogy(supports, errors_1, '.-', label='1, uniform')
plt.scatter(supports[fails_1], errors_1[fails_1], s=75, c='black', marker='x')
plt.semilogy(supports, errors_2_u, '.-', label=r'$2\times2$, uniform')
plt.scatter(supports[fails_2_u], errors_2_u[fails_2_u], s=75, c='black', marker='x')
plt.semilogy(supports, errors_2_g, '.-', label=r'$2\times2$, Gauss-Legendre')
plt.scatter(supports[fails_2_g], errors_2_g[fails_2_g], s=75, c='black', marker='x')
plt.semilogy(supports, errors_3_u, '.-', label=r'$3\times3$, uniform')
plt.scatter(supports[fails_3_u], errors_3_u[fails_3_u], s=75, c='black', marker='x')
plt.xticks([1.0, 1.5, 2.0, 2.5])
# plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel(r'$E_2$ magnitude')
# plt.title('Error Magnitudes')
# plt.legend(fontsize='x-large')

# plot the times
plt.subplot(132)
plt.plot(supports, times_1, '.-', label='1, uniform')
plt.scatter(supports[fails_1], times_1[fails_1], s=75, c='black', marker='x')
plt.plot(supports, times_2_u, '.-', label=r'$2\times2$, uniform')
plt.scatter(supports[fails_2_u], times_2_u[fails_2_u], s=75, c='black', marker='x')
plt.plot(supports, times_2_g, '.-', label=r'$2\times2$, Gauss-Legendre')
plt.scatter(supports[fails_2_g], times_2_g[fails_2_g], s=75, c='black', marker='x')
plt.plot(supports, times_3_u, '.-', label=r'$3\times3$, uniform')
plt.scatter(supports[fails_3_u], times_3_u[fails_3_u], s=75, c='black', marker='x')
plt.ylim([0, 15])
plt.yticks(np.linspace(0, 15, 7))
plt.xticks([1.0, 1.5, 2.0, 2.5])
plt.xlabel(r'support size $d/h$')
plt.ylabel('computation time [s]')
# plt.title('Computation Times')
# plt.legend(fontsize='x-large', loc='upper left')

# plot the condition number of the stiffness matrix
plt.subplot(133)
plt.semilogy(supports, conds_1, '.-', label='1, uniform')
plt.scatter(supports[fails_1], conds_1[fails_1], s=75, c='black', marker='x')
plt.semilogy(supports, conds_2_u, '.-', label=r'$2\times2$, uniform')
plt.scatter(supports[fails_2_u], conds_2_u[fails_2_u], s=75, c='black', marker='x')
plt.semilogy(supports, conds_2_g, '.-', label=r'$2\times2$, Gauss-Legendre')
plt.scatter(supports[fails_2_g], conds_2_g[fails_2_g], s=75, c='black', marker='x')
plt.semilogy(supports, conds_3_u, '.-', label=r'$3\times3$, uniform')
plt.scatter(supports[fails_3_u], conds_3_u[fails_3_u], s=75, c='black', marker='x')
plt.xticks([1.0, 1.5, 2.0, 2.5])
# plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel('Frobenius norm condition number')
# plt.title('Condition Numbers')
plt.legend(loc='upper left', bbox_to_anchor=(0,0.9))

# plt.savefig(f"MLS_timings_quadrature_1k.pdf",
#     bbox_inches = 'tight', pad_inches = 0)