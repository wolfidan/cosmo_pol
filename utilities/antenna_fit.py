# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 11:21:17 2016

@author: wolfensb
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import argrelextrema
import pickle



def gaussian_sum(x,params):
    return 10*np.log10(np.sum([10**(0.1*p[0])*np.exp(-(x-p[1])**2/(2*p[2]**2)) for p in params],axis=0))
    
def obj(params,x,y):
    params = np.reshape(params,(len(params)/3,3))
    est=gaussian_sum(x,params)
    
    error=np.sqrt(np.sum((est-y)**2))
    
    return error


def antenna_diagram(x,bw,ampl,mu):
    sigma = bw/(2*np.sqrt(2*np.log(2)))
    y = np.zeros(x.shape)
    for i in range(len(bw)):
        y = y+10**(0.1*ampl[i])*np.exp(-(x-mu[i])**2/(2*sigma[i]**2))
    return y
    


def optimize_gaussians(x,y,n_lobes):
    peaks = argrelextrema(y, np.greater)
    a_lobes = y[peaks]
    mu_lobes = x[peaks]
    
    if 0 not in mu_lobes:
        mu_lobes=np.append(mu_lobes,0)
        a_lobes=np.append(a_lobes,0)
    
    params=np.column_stack((a_lobes,mu_lobes))
    params=params[params[:,0].argsort()]
    params=np.flipud(params)
    
    selected = params[0:n_lobes,:]

    p0=np.column_stack((selected[:,0],selected[:,1],np.array([0.5]*n_lobes)))
    
    bounds=[]
    for i in range(n_lobes):
        for j in range(3):
            if j!=2:
                bounds.append([None,None])
            else:
                bounds.append([0.1,2])
           
#    bounds[0]=[0,0]
#    bounds[1]=[0,0]

    cons={'type':'eq','fun':lambda x: gaussian_sum(0,np.reshape(x,(len(x)/3,3)))}
    params = minimize(obj,p0,args=(x,y),bounds=bounds,method='SLSQP',constraints=cons)
    params = np.reshape(params['x'],(n_lobes,3))
    return params

if __name__ == '__main__':
    data = np.genfromtxt('/media/wolfensb/Storage/cosmo_pol/tests/antenna.csv',delimiter=',')
    x = data[:,0]
    y = 10*np.log10((10**(0.1*data[:,1]))**2)

    p1 = optimize_gaussians(x,y,5)
    p2 = optimize_gaussians(x,y,7)
    
    plt.figure()
    plt.plot(x,y,linewidth=2)
    plt.hold(True)
    plt.plot(x,gaussian_sum(x,p1),linewidth=2)
    plt.plot(x,gaussian_sum(x,p2),linewidth=2)
    sigma=1./np.sqrt(2)*1.45/(2*np.sqrt(2*np.log(2)))
    plt.plot(x,gaussian_sum(x,[[0,0,sigma]]),linewidth=2)
    plt.grid()
    plt.legend(['Real antenna','5 gaussians', '7 gaussians','3dB gaussian'],loc=0)
    plt.ylim(-100,0)
    plt.savefig('ex_fit_0deg.pdf',dpi=200,bbox_inches='tight')
    
    
        
    #plt.plot(x,y,x,10*np.log10(10**(0.1*-35)*np.exp(-x**2/(2*5**2))))
    #
    #ang = np.linspace(-14,14,1000)
    #
    #bw = [5,1.5,1.5,1.8,1.5,1.5,6]
    #sigma = sigma = bw/(2*np.sqrt(2*np.log(2)))
    #mu = [-10,-7,-4,0,4,7,10]
    #a_dB = [-40,-35,-30,0,-28,-40,-43]
    #
    #
    #plt.figure()
    #plt.plot(antenna['angles'],10*np.log10(d),ang,10*np.log10(antenna_diagram(ang,bw,a_dB,mu)),ang,10*np.log10(antenna_diagram(ang,[1.45],[0],[0])))
    #plt.ylim([-50,0])
    #plt.xlabel('Angle [$^{\circ}$]')
    #plt.ylabel('Normalized power [dB]')
    #plt.grid()
    #plt.legend(['Real','Sum of gaussians','Single gaussian'])
    #plt.savefig('test_antenna_gauss.pdf',bbox_inches='tight')
    #
    #
    #h = lambda x,y : np.sqrt(x**2+y**2)**0.3
    # 
    ##h = lambda r,theta : 1
    #
    #integ = 0
    #sum_weights = 0
    #for i in range(NPTS_LEG):
    #    for j in range(len(bw)):
    #        for k in range(NPTS_HER):
    #            r = (mu[j]+np.sqrt(2)*sigma[j]*x_her[k])
    #            weight = np.pi/2 * w_leg[i] * w_her[k]*10**(0.1*a_dB[j])*(np.sqrt(2)*sigma[j])*abs(r)
    #            theta = np.pi/2*x_leg[i]+np.pi/2
    #            integ+=weight*h(r*np.cos(theta),r*np.sin(theta))
    #            sum_weights += weight
    #print integ/sum_weights
    #
    #
    #theta = np.linspace(0,2*np.pi,1000)
    #
    #r,t = np.meshgrid(ang,theta)
    #
    #f = antenna_diagram(r,bw,a_dB,mu)
    #print np.sum(f*h(r*np.cos(t),r*np.sin(t)))/np.sum(f)
    #
    #
    #x,y = np.meshgrid(antenna['angles'],antenna['angles'])
    #print np.sum(antenna['data']*h(x,y))/np.sum(antenna['data'])
    ##
    #def func(x, a, b, c,d,e,f,g,h,i):
    #    return a * np.exp(-(x-b)**2/(2*c**2))+ d * np.exp(-(x-e)**2/(2*f**2)) + g * np.exp(-(x-h)**2/(2*i**2))
    #
    #from scipy.optimize import curve_fit
    #popt, pcov = curve_fit(func,antenna['angles'], d,p0=[1,0,0.7,0.001,-4,0.7,0.001,4,0.7])
    #
    #plt.plot(antenna['angles'],10*np.log10(d),antenna['angles'],10*np.log10(func(antenna['angles'],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8])))
    #plt.ylim([-50,0])
    #
