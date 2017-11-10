# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:53:48 2015

@author: wolfensb
"""

import numpy as np
#import scipy.spatial.qhull as qhull
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


from cosmo_pol.constants import constants
from cosmo_pol.utilities import cfg

def binary_search(t, x):
    ind = t.searchsorted(x)
    if ind == len(t):
        return ind - 1 # x > max(t)
    elif ind == 0:
        return 0
    before = ind-1
    if x-t[before] < t[ind]-x:
        ind -= 1
    return ind

def vlinspace(a, b, N, endpoint=True):
    a, b = np.asanyarray(a), np.asanyarray(b)
    return a[..., None] + (b-a)[..., None]/(N-endpoint) * np.arange(N)

def nan_cumprod(x):
    x[np.isnan(x)]=1
    return np.cumprod(x)

def nan_cumsum(x):
    x[np.isnan(x)]=0
    return np.cumsum(x)

def sum_arr(x,y, cst = 0):
    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))
            pad_1.append((0,0))


    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)

    z = np.sum([x,y],axis=0)

    return z

def nansum_arr(x,y, cst = 0):

    x = np.array(x)
    y = np.array(y)

    diff = np.array(x.shape) - np.array(y.shape)
    pad_1 = []
    pad_2 = []
    for d in diff:
        if d < 0:
            pad_1.append((0,-d))
            pad_2.append((0,0))
        else:
            pad_2.append((0,d))
            pad_1.append((0,0))

    x = np.pad(x,pad_1,'constant',constant_values=cst)
    y = np.pad(y,pad_2,'constant',constant_values=cst)

    z = np.nansum([x,y],axis=0)
    return z

def cut_at_sensitivity(list_beams):

    sens_config = cfg.CONFIG['radar']['sensitivity']
    if not isinstance(sens_config,list):
        sens_config = [sens_config]

    if len(sens_config) == 3: # Sensitivity - gain - snr
        threshold_func = lambda r: sens_config[0] + constants.RADAR_CONSTANT_DB \
                         + sens_config[2] + 20*np.log10(r/1000.)
    elif len(sens_config) == 2: # ZH - range
        threshold_func = lambda r: (sens_config[0] - 20*np.log10(sens_config[1]/1000.)) \
                                + 20*np.log10(r/1000.)
    elif len(sens_config) == 1: # ZH
        threshold_func = lambda r: sens_config[0]

    else:
        print('Sensitivity parameters are invalid, cannot cut at specified sensitivity')
        print('Enter either a single value of refl (dBZ) or a list of',
              '[refl (dBZ), distance (m)] or a list of [sensitivity (dBm), gain (dBm) and snr (dB)]')

        return list_beams

    if isinstance(list_beams[0],list): # Loop on list of lists
        for i,sweep in enumerate(list_beams):
            for j,b, in enumerate(sweep):
                rranges = (cfg.CONFIG['radar']['radial_resolution'] *
                           np.arange(len(b.dist_profile)))
                mask = 10*np.log10(b.values['ZH']) < threshold_func(rranges)
                for k in b.values.keys():
                    if k in constants.SIMULATED_VARIABLES:
                        if k == 'DSPECTRUM':
                            logspectrum = 10*np.log10(list_beams[i][j].values[k])
                            thresh = threshold_func(rranges)
                            thresh =  np.tile(thresh,(logspectrum.shape[1],1)).T
                            list_beams[i][j].values[k][logspectrum < thresh] = np.nan
                        else:
                            list_beams[i][j].values[k][mask] = np.nan

    else:
        for i,b in enumerate(list_beams): # Loop on simple list
            rranges = cfg.CONFIG['radar']['radial_resolution'] * np.arange(len(b.dist_profile))
            mask = 10*np.log10(b.values['ZH']) < threshold_func(rranges)
            for k in b.values.keys():
                if k in constants.SIMULATED_VARIABLES:
                    list_beams[i].values[k][mask] = np.nan
    return list_beams



def combine_beams(list_of_beams):
    x = list_of_beams[0]
    for l in list_of_beams:
        if (l.dist_profile==x.dist_profile).all():
            x.values.update(l.values)
        else:
            print 'Beams are not defined on the same set of coordinates, aborting'
            return
    return x

def divide_by_power(sign,power):
    out = sign/power
    out[np.isinf(out)] = np.nan
    return out

def nice_histogram(dataset,n_bins = [], plot_style = 'stairs'):
    if plot_style not in ['stairs','lines']:
        print("Invalid plot_style, use either 'stairs' or 'lines'")
        print('Using stairs instead')
        plot_style = 'stairs'

    n_datasets=len(dataset)

    if n_datasets>5:
        print('This function supports only 5 arrays at max, sorry')
        return

    if np.isscalar(n_bins):
        n_bins = [n_bins for d in dataset]

    if n_bins != [] and len(n_bins) != len(dataset):
        print('Length of n_bins does not correspond to length of dataset',
              ', taking default values instead')

        n_bins = []

    colors=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
    xlims=[np.Inf,-np.Inf]
    ylims=[np.Inf,-np.Inf]
    for i in range(n_datasets):
        col=colors[i]
        data = dataset[i]
        data = data[np.isfinite(data)]

        if n_bins != []:
            hist, bin_edges = np.histogram(data, density=True, bins = n_bins[i])
        else:
            hist, bin_edges = np.histogram(data, density=True)

        x = 0.5 * (bin_edges[0:-1] + bin_edges[1:])
        y = hist

        if plot_style == 'stairs':
            plt.fill_between(x,0,hist, alpha=0.5,color=col,step='mid',label='_nolegend_')
            plt.step(x,hist,linewidth=2,color=col,where='mid')
        elif plot_style == 'lines':
            plt.fill_between(x,0,hist, alpha=0.5,label='_nolegend_',color=col)
            plt.plot(x,hist,linewidth=2,color=col)

        xlims=[np.min([np.min(x*0.9),xlims[0]]),np.max([np.max(x*1.1),xlims[1]])]
        ylims=[np.min([np.min(y*0.9),ylims[0]]),np.max([np.max(y*1.1),ylims[1]])]

    plt.xlim(xlims)
    plt.ylim(ylims)
    return

def polar2cartesian(r, t, grid, x, y, order=3):

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(X, Y)+np.pi

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_it, new_ir]),
                            order=order).reshape(new_r.shape)

def vector_1d_to_polar(angles, values, x,y):
    midpt = int(np.floor(len(angles)/2))
    r = angles[midpt:]
    thet = [0,np.pi,2*np.pi]

    pol=np.zeros((len(thet),len(r)))
    pol[0,:]=values[midpt:]
    pol[1,:]=values[0:midpt+1]
    pol[1,:]=pol[1,::-1]
    pol[2,:]=pol[0,:]

    return  polar2cartesian(r,thet,pol,x,y)

if __name__ == '__main__':
#    angles = np.arange(-15,16,1)
#    vals = np.exp(-0.001*angles**2)
#    vals[0:16]=np.exp(-0.001*angles[0:16]**2)
#    vals[15:]=np.exp(-0.01*angles[15:]**2)
#
#    bounds = 15
#    pts_hor, weights_hor=np.polynomial.legendre.leggauss(5)
#    pts_hor=pts_hor*bounds
#
#    pts_ver, weights_ver=np.polynomial.legendre.leggauss(9)
#    pts_ver=pts_ver*bounds
#
#    uu = vector_1d_to_polar(angles,vals,pts_hor,pts_ver).T
#
#    plt.imshow(uu)


    a = np.random.randn(10000,)

    b = np.random.randn(10000,)

    nice_histogram([a,b],n_bins = [100,100])

#def fast_regridding_weights(img, zoom):
#    N,M=img.shape
#    o_grid=np.meshgrid(range(0,N),range(0,M))
#    i_grid=np.meshgrid(np.linspace(0,N,zoom*N),np.linspace(0,M,zoom*M))
#    return interp_weights(o_grid, i_grid)
#
#def interp_weights(xy,XY):
#
#    XY_vec=np.vstack((XY[0].ravel(),XY[1].ravel())).T
#    xy_vec=np.vstack((xy[0].ravel(),xy[1].ravel())).T
#    d=xy_vec.shape[1]
#    tri = qhull.Delaunay(xy_vec)
#    simplex = tri.find_simplex(XY_vec)
#    vertices = np.take(tri.simplices, simplex, axis=0)
#    temp = np.take(tri.transform, simplex, axis=0)
#    delta = XY_vec - temp[:, d]
#    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
#
#    output={}
#    output['bary']=np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
#    output['vertices']=vertices
#    output['shape']=XY[0].shape
#    return output
#
#def interpolate(values, w, fill_value=np.nan):
#    if not isinstance(values, list):
#        values=values.ravel()
#
#    vtx=w['vertices']
#    wts=w['bary']
#    values_interp=np.einsum('nj,nj->n', np.take(values, vtx), wts)
#    values_interp[np.any(wts < 0, axis=1)] = fill_value
#    return np.reshape(values_interp,w['shape'])
#
#from tictoc import *
#
#x=np.random.rand(350,520)
#tic()
#w=fast_regridding_weights(x,4)
#for i in range(60):
#    o=interpolate(x,w)
#toc()