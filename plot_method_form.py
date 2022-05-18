#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:37:36 2020

@author: Samuel A. Maloney
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

supports_G = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3,
        2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5])
errors_G_c = np.array([7.73357916e-04, 7.30865288e-04, 6.56931373e-04, 5.75342982e-04,
       5.18137458e-04, 5.07906014e-04, 4.29218489e-04, 3.07947647e-04,
       2.55740236e-04, 3.35761055e-04, 5.42439666e-04, 8.46176310e-04,
       1.24205793e-03, 1.70241900e-03, 2.20424614e-03, 2.77800114e-03,
       3.52832190e-03, 4.58693251e-03, 6.16164561e-03, 8.59901150e-03,
       1.30339107e-02, 3.12235214e-02, 3.43141475e-02, 6.45299089e-02,
       1.64901907e-01])
times_G_c = np.array([  2.5479633,   2.5045587,   2.487235 ,   2.5094735,   2.5042793,
         2.9206844,   2.8838649,   2.9521906,   2.9944942,   2.9300401,
         2.9197826,   3.0311141,   3.7103014,   3.2237634,   3.2370239,
         3.7413433,   3.8826879,   3.9634975,   3.859536 ,   4.476024 ,
         4.5547081, 110.5788926,   4.8140588,   4.7837177,   4.9553713])
conds_G_c = np.array([4.48100298e+03, 4.48100247e+03, 4.48100305e+03, 4.48100310e+03,
       4.48100313e+03, 4.48102224e+03, 4.48109918e+03, 4.48106172e+03,
       4.48105416e+03, 4.48105786e+03, 4.48108461e+03, 4.48123117e+03,
       4.48211520e+03, 4.48927799e+03, 4.58528242e+03, 4.94788515e+03,
       2.10160210e+05, 1.40884411e+07, 1.46166172e+04, 1.44986774e+04,
       7.38143483e+04, 4.87552773e+12, 1.65027273e+06, 1.71525926e+04,
       7.86909352e+04])
fails_G_c = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False])
# ##### Laptop, No Preconditioner #####
# times_G_c = np.array([ 2.5991813,  2.5081271,  2.5394546,  2.6040817,  2.5735656,
#         2.8886825,  2.976716 ,  2.8481989,  3.0671982,  2.9035487,
#         3.1430624,  3.3660009,  3.8924743,  4.520682 ,  6.3437983,
#        17.8406575, 20.3681388, 20.2306733, 20.5588163, 25.5196647,
#        25.2292667, 25.7270667, 25.739674 , 25.7081187, 25.6093098])
# ##### Lab Desktop, No Preconditioner #####
# errors_G_c = np.array([7.73365617e-04, 7.30868645e-04, 6.56948039e-04, 5.75361761e-04,
#         5.18151051e-04, 5.07897017e-04, 4.29229371e-04, 3.07949936e-04,
#         2.55732775e-04, 3.35771771e-04, 5.42441322e-04, 8.46177611e-04,
#         1.24205809e-03, 1.70241828e-03, 2.20424607e-03, 2.77800109e-03,
#         3.52809077e-03, 4.58681837e-03, 6.15766257e-03, 8.57736705e-03,
#         1.26551488e-02, 1.99823519e-02, 3.35911277e-02, 6.39707033e-02,
#         1.45740477e-01])
# times_G_c = np.array([ 2.55538883,  2.53415708,  2.51122443,  2.59759859,  2.58615548,
#         2.65314941,  2.62434362,  2.61950575,  2.68502776,  2.69572755,
#         2.77162035,  2.98910332,  3.28325016,  3.95279311,  5.11618559,
#         12.1202988 , 15.14076699, 15.14265283, 15.12104725, 19.07310229,
#         19.56369004, 18.91503815, 18.91719603, 19.01263955, 18.85876551])
# conds_G_c = np.array([5.54880890e+04, 5.54751937e+04, 5.54538938e+04, 5.54325332e+04,
#         5.54208695e+04, 5.53702182e+04, 5.46122150e+04, 5.77386573e+04,
#         7.04570680e+04, 1.00626866e+05, 1.69140936e+05, 3.33915017e+05,
#         7.35718877e+05, 1.88800734e+06, 6.36342543e+06, 3.93743794e+07,
#         4.24331320e+08, 6.59020073e+08, 1.25224315e+08, 6.68840568e+07,
#         1.49154938e+08, 1.49912212e+09, 6.55408162e+08, 9.20755367e+07,
#         1.41141356e+08])
# fails_G_c = np.array([False, False, False, False, False, False, False, False, False,
#         False, False, False, False, False, False, False,  True,  True,
#         True,  True,  True,  True,  True,  True,  True])

errors_G_q = np.array([7.66028854e-04, 6.95682925e-04, 5.94489071e-04, 5.19086019e-04,
       5.15959626e-04, 5.73608197e-04, 3.97704227e-04, 2.21269581e-04,
       4.13270367e-04, 7.50718371e-04, 1.25765144e-03, 1.94265792e-03,
       2.78503257e-03, 3.89516863e-03, 5.56634231e-03, 9.35243379e-03,
       2.09297278e-02, 3.15948056e-02, 3.15427351e-02, 5.72437134e-02,
       8.91007834e-02, 9.41282719e-02, 8.31888477e-02, 7.21600698e-02,
       6.45993746e-02])
times_G_q = np.array([2.5913601, 2.504637 , 2.4232433, 2.4491324, 2.467374 , 2.7759878,
       2.7765234, 2.7762038, 2.8741386, 3.2306686, 2.8919891, 3.2031938,
       3.1081602, 3.1755585, 3.2372723, 3.6399066, 3.737464 , 3.7804739,
       3.794562 , 4.249632 , 4.6226727, 4.5924069, 4.6138759, 4.763917 ,
       4.7182814])
conds_G_q = np.array([4.48100300e+03, 4.48100304e+03, 4.48100309e+03, 4.48100221e+03,
       4.48100289e+03, 4.48102895e+03, 4.48106045e+03, 4.48105532e+03,
       4.48107184e+03, 4.48134492e+03, 4.48612232e+03, 4.01965056e+05,
       5.23239979e+03, 5.05999528e+03, 5.43943990e+03, 4.72681460e+03,
       6.96535703e+03, 1.05744685e+04, 4.56788503e+03, 4.69691219e+03,
       4.56439648e+03, 4.71852248e+03, 5.27849889e+03, 5.43918381e+03,
       5.12029190e+03])
fails_G_q = np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False])
# ##### Laptop, No Preconditioner #####
# times_G_q = np.array([ 2.5091905,  2.4935234,  2.5431507,  2.5438491,  2.5210946,
#         3.0237127,  2.8008514,  2.804802 ,  2.8042468,  3.1401288,
#         3.8631995, 14.1216238,  8.4478961,  9.4121674, 12.1224801,
#         20.5427678, 22.0403895, 20.3225956, 20.4742536, 25.2526122,
#         25.1888478, 25.6293119, 25.7281078, 25.8524326, 25.903135 ])
# conds_G_q = np.array([5.54858301e+04, 5.54648641e+04, 5.54372728e+04, 5.54209769e+04,
#         5.54319931e+04, 5.53839735e+04, 5.52489670e+04, 7.02311069e+04,
#         1.31415775e+05, 3.71569333e+05, 2.28296656e+06, 3.28932266e+08,
#         1.37637776e+07, 9.52833418e+06, 1.08989293e+07, 1.70198267e+07,
#         6.36593427e+07, 1.07127202e+08, 9.02837311e+06, 1.16860154e+07,
#         9.06547675e+06, 1.04893915e+07, 1.92904093e+07, 2.21314244e+07,
#         1.97079788e+07])
# fails_G_q = np.array([False, False, False, False, False, False, False, False, False,
#         False, False,  True, False, False, False,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True])
# ##### Desktop, No Preconditioner #####
# times_G_q = np.array([ 2.57291831,  2.60908935,  2.61327973,  2.60607194,  2.66251062,
#         2.74942154,  2.76613232,  2.76711235,  2.76992798,  2.86991612,
#         3.4698109 , 10.67953857,  7.06864877,  7.30408401,  9.43283859,
#        15.32809334, 15.32209693, 15.3201919 , 15.23474473, 18.83929415,
#        19.00364715, 19.08652072, 19.3051175 , 19.20002792, 19.48456151])

supports_C = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
       2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5])
errors_C_c = np.array([3.99442080e-04, 4.45414034e-04, 5.38317716e-04, 6.70478980e-04,
       6.23421719e-04, 6.16639062e-04, 6.86612039e-04, 7.88424549e-04,
       8.98382490e-04, 1.00543349e-03, 5.97290451e-04, 4.90287369e-04,
       1.48540283e-04, 8.94966868e-05, 3.77745013e-05, 1.63142154e-04,
       4.10318069e-04, 6.90785499e-04, 8.43396405e-04, 9.56870767e-04,
       5.03439690e-04, 1.11850846e-04, 7.60141408e-04, 9.99829844e-04,
       9.53401498e-04])
times_C_c = np.array([ 9.8341885,  4.174162 ,  4.163808 ,  4.2785016,  4.582523 ,
        4.5038046,  4.4470526,  4.4747481,  4.5370393,  4.6627305,
        4.6961305,  4.637851 ,  5.2125058,  5.292229 ,  5.4016876,
        5.8020948,  8.753225 , 10.9470429,  9.5391368, 10.2428875,
        7.6572595, 16.3338001, 16.1156611, 16.2285698, 16.2549509])
conds_C_c = np.array([7.82822352e+04, 7.82824474e+04, 7.82842349e+04, 7.82900005e+04,
       6.94928164e+04, 6.11672413e+04, 5.61192968e+04, 5.32070683e+04,
       5.14689745e+04, 5.03106748e+04, 4.03506786e+04, 3.43465956e+04,
       2.64154108e+04, 2.14320383e+04, 2.03618744e+04, 1.94797233e+05,
       2.15686237e+05, 3.67530081e+05, 3.03448501e+05, 4.18078336e+05,
       1.02857119e+06, 4.42660480e+06, 9.44202493e+06, 3.64710895e+06,
       1.27566055e+06])
fails_C_c = np.array([ True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True,  True,  True,  True])
# ##### No Preconditioner #####
# times_C_c = np.array([ 8.2576872,  8.2377022,  8.2485965,  8.1871661,  8.7366865,
#         8.7843262, 10.8607179,  8.8814676,  8.7787057,  8.8259171,
#        10.6065565,  9.55416  , 11.3109812, 11.091552 , 11.0816281,
#        11.0067451, 11.1014563, 11.0964052, 12.2505379, 12.0720346,
#        12.98341  , 14.3193531, 15.0989452, 15.1311408, 14.4631419])
# conds_C_c = np.array([6.33123188e+08, 2.72768430e+08, 1.52457389e+08, 9.74732062e+07,
#        6.98259983e+07, 5.02987495e+07, 3.74818483e+07, 2.97220965e+07,
#        2.52278076e+07, 2.27663974e+07, 2.04650341e+07, 1.96125027e+07,
#        2.04085427e+07, 2.19194778e+07, 2.52175248e+07, 1.12280527e+08,
#        3.43363280e+08, 6.16171385e+08, 5.46340440e+08, 1.14604069e+09,
#        2.48696728e+09, 4.59504167e+09, 2.73889632e+10, 1.20572569e+10,
#        4.01078694e+09])
# fails_C_c = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True])

errors_C_q = np.array([4.07025278e-04, 4.87402275e-04, 6.35049104e-04, 8.21945460e-04,
       6.63195593e-04, 6.71363715e-04, 7.62700516e-04, 8.69432146e-04,
       9.72067033e-04, 1.06388903e-03, 3.20661822e-04, 3.08483011e-04,
       4.65852537e-04, 6.28678086e-04, 4.67506147e-04, 1.28512678e-04,
       3.36810200e-04, 7.96629894e-04, 1.04366296e-03, 1.23313824e-03,
       4.06360237e-04, 4.79721803e-04, 1.23242184e-03, 1.37653556e-03,
       1.11647410e-03])
times_C_q = np.array([9.856784 , 4.2878018, 4.2958137, 4.2746453, 4.5230437, 4.5536526,
       4.5032071, 4.5008888, 4.4951109, 4.4547155, 4.7970063, 5.0117772,
       5.4173794, 5.5107796, 5.2602706, 5.2797644, 5.2750456, 5.7289462,
       5.6314053, 5.6227668, 6.3222715, 6.8086253, 6.5336368, 6.437217 ,
       6.3318567])
conds_C_q = np.array([7.82822383e+04, 7.82830326e+04, 7.82880611e+04, 7.83019833e+04,
       6.20548234e+04, 5.21237161e+04, 4.74998648e+04, 4.53147957e+04,
       4.41363453e+04, 4.33247068e+04, 3.26808864e+04, 7.42969231e+04,
       6.47825287e+04, 1.72632795e+04, 1.56710587e+04, 1.57934477e+04,
       1.82131154e+04, 1.61029494e+05, 3.70383132e+04, 2.63152438e+04,
       9.02200137e+04, 8.41546873e+04, 2.33701440e+04, 2.39329025e+04,
       4.74942015e+04])
fails_C_q = np.array([ True, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False])
# ##### No Preconditioner #####
# times_C_q = np.array([ 8.3934474,  8.2640156,  8.1855235,  8.2745329,  8.8033927,
#         8.7764313,  8.8106165,  8.830471 ,  8.8655013,  8.7895667,
#         9.7849107,  9.6836806, 11.1219282, 11.1820776, 11.1388422,
#        11.1171752, 11.5742983, 11.1996252,  5.1390923,  5.3158393,
#        12.8553766, 14.3352505,  5.7782566,  5.7757499,  7.5337894])
# conds_C_q = np.array([5.83381804e+08, 2.19895883e+08, 1.09880541e+08, 6.70320041e+07,
#        4.90995160e+07, 3.54051523e+07, 2.70304185e+07, 2.26381431e+07,
#        2.06753977e+07, 2.02292368e+07, 2.22174203e+07, 1.37357636e+08,
#        8.39202472e+07, 2.91962638e+07, 3.05688390e+07, 3.32413281e+07,
#        3.75668297e+07, 1.29282510e+08, 5.29554027e+07, 5.43935267e+07,
#        1.58681421e+08, 1.49639240e+08, 4.92007782e+07, 4.69006134e+07,
#        5.00934986e+07])
# fails_C_q = np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True,  True,  True,  True,  True,  True,  True,  True,  True,
#        False, False,  True,  True, False, False, False])

##### Old Data #####
# supports_C = np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
#         2.8, 2.9, 3. ])
# errors_C_c = np.array([8.49958537E-01, 5.58828566E-01, 3.77868646E-01, 2.53769047E-01,
#         8.24589745E-02, 8.67944210E-02, 1.12225947E-01, 2.36910926E-01,
#         3.32407931E-01, 3.40898375E-01, 8.80199887E-02, 1.42015683E-01,
#         1.68331685E-01, 2.11418793E-01, 3.34909694E-01, 2.43119918E-01])
# times_C_c = np.array([7.28704831, 7.19890174, 7.80368475, 7.33962689, 3.19004136,
#         3.13762179, 3.70495732, 7.63920335, 8.82569128, 8.78560604,
#         3.87287002, 8.774823  , 8.76574467, 8.79760845, 9.44451105,
#         9.4731756 ])
# conds_C_c = np.array([2.33759289e+09, 1.68535670e+08, 1.16450373e+08, 1.72815424e+08,
#         2.44446204e+07, 2.21953273e+07, 2.01414617e+07, 2.33863166e+07,
#         3.07024466e+07, 2.51882168e+07, 2.51844050e+07, 1.12958576e+08,
#         4.05262184e+08, 2.18779789e+09, 6.42884362e+08, 1.02697984e+09])
# fails_C_c = np.array([ True,  True,  True,  True, False, False, False,  True,  True,
#         True, False,  True,  True,  True,  True,  True])

# errors_C_q = np.array([5.70576451E-01, 2.58221383E-01, 1.11047708E-01, 8.64858201E-02,
#         1.32614987E-01, 2.19299108E-01, 3.15817153E-01, 4.13913451E-01,
#         5.08345359E-01, 5.91870651E-01, 5.56528007E-01, 2.38666126E-01,
#         3.01908135E-01, 4.07124081E-01, 2.92438342E-01, 8.13446863E-01])
# times_C_q = np.array([7.2708704 , 7.24994387, 3.03880449, 3.0322526 , 3.0329999 ,
#         7.16458348, 7.79776608, 7.79655113, 8.9454295 , 9.07199307,
#         8.86134529, 3.78839149, 3.81958129, 8.87147831, 4.70290467,
#         9.57286041])
# conds_C_q = np.array([2.17538265E+09, 6.83135902E+07, 2.61663710E+07, 2.23589068E+07,
#         2.03911928E+07, 1.83524227E+08, 7.37481918E+07, 3.40220759E+08,
#         1.48351025E+09, 3.16348384E+07, 3.17774659E+07, 3.33008113E+07,
#         3.79620635E+07, 1.19597543E+08, 5.30241505E+07, 5.44961818E+07])
# fails_C_q = np.array([ True,  True, False, False, False,  True,  True,  True,  True,
#         True,  True, False, False,  True, False,  True])
##### End Old Data #####


##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,2.5)
plt.subplots_adjust(wspace = 0.35)

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot the minimum E_2 error norms
plt.subplot(131)
plt.semilogy(supports_G, errors_G_c, '.-', label='Galerkin, cubic')
plt.scatter(supports_G[fails_G_c], errors_G_c[fails_G_c], s=75, c='black', marker='x')
plt.semilogy(supports_G, errors_G_q, '.-', label='Galerkin, quartic')
plt.scatter(supports_G[fails_G_q], errors_G_q[fails_G_q], s=75, c='black', marker='x')
plt.semilogy(supports_C, errors_C_c, '.-', label='Collocation, cubic')
plt.scatter(supports_C[fails_C_c], errors_C_c[fails_C_c], s=75, c='black', marker='x')
plt.semilogy(supports_C, errors_C_q, '.-', label='Collocation, quartic')
plt.scatter(supports_C[fails_C_q], errors_C_q[fails_C_q], s=75, c='black', marker='x')
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
# plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel(r'$E_2$ error norm')
# plt.title('Error Magnitudes')
# plt.legend(fontsize='x-large')

# plot the times
plt.subplot(132)
plt.plot(supports_G, times_G_c, '.-', label='Galerkin, cubic')
plt.scatter(supports_G[fails_G_c], times_G_c[fails_G_c], s=75, c='black', marker='x')
plt.plot(supports_G, times_G_q, '.-', label='Galerkin, quartic')
plt.scatter(supports_G[fails_G_q], times_G_q[fails_G_q], s=75, c='black', marker='x')
plt.plot(supports_C, times_C_c, '.-', label='collocation, cubic')
plt.scatter(supports_C[fails_C_c], times_C_c[fails_C_c], s=75, c='black', marker='x')
plt.plot(supports_C, times_C_q, '.-', label='collocation, quartic')
plt.scatter(supports_C[fails_C_q], times_C_q[fails_C_q], s=75, c='black', marker='x')
plt.ylim([0, 18])
plt.yticks(np.linspace(0, 17.5, 8))
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
plt.xlabel(r'support size $d/h$')
plt.ylabel('computation time [s]')
# plt.title('Computation Times')
plt.legend(loc='upper left')

# plot the condition number of the stiffness matrix
plt.subplot(133)
plt.semilogy(supports_G, conds_G_c, '.-', label='Galerkin, cubic')
plt.scatter(supports_G[fails_G_c], conds_G_c[fails_G_c], s=75, c='black', marker='x')
plt.semilogy(supports_G, conds_G_q, '.-', label='Galerkin, quartic')
plt.scatter(supports_G[fails_G_q], conds_G_q[fails_G_q], s=75, c='black', marker='x')
plt.semilogy(supports_C, conds_C_c, '.-', label='Collocation, cubic')
plt.scatter(supports_C[fails_C_c], conds_C_c[fails_C_c], s=75, c='black', marker='x')
plt.semilogy(supports_C, conds_C_q, '.-', label='Collocation, quartic')
plt.scatter(supports_C[fails_C_q], conds_C_q[fails_C_q], s=75, c='black', marker='x')
plt.ylim([3e3, 3e7])
plt.xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
# plt.xlabel('support size [multiple of grid spacing]')
plt.ylabel('Frobenius norm condition number')
# plt.title('Condition Numbers')
# plt.legend(fontsize='x-large')

# plt.savefig(f"MLS_timings_method_form_1k.pdf",
#     bbox_inches = 'tight', pad_inches = 0)