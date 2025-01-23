#%%#%%
import sys, os
os.chdir('/mnt/home/cfilion/DREAMS/')
#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import KDTree
import pynbody
from scipy import stats
from matplotlib.colors import LogNorm
sys.path.append('/mnt/home/cfilion/DREAMS/EXP-tools/')
sys.path.append('/mnt/home/cfilion/DREAMS/EXP/lib/')
sys.path.append('/mnt/home/cfilion/DREAMS/EXP-tools/EXPtools/')
import pyEXP
from basis_builder import makemodel
from inspect import getmembers, isfunction
from reader_funcs import *
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit, least_squares
####
#### some nice functions to do analysis of our DREAMS disks!
####
def sech2(x):
    return 1/np.cosh(x)**2

def half_mass(snapshot, verbosity = 3):
    '''compute the half-mass radius from a simulation snapshot'''
    mass = snapshot['mass']
    R = np.sqrt(snapshot['x']**2 + snapshot['y']**2 + snapshot['z']**2)
    cumulative_mass = np.cumsum(mass[np.argsort(R)])
    half_mass = np.max(cumulative_mass)/2
    half_mass_idx = np.argmin(np.abs(cumulative_mass - half_mass))
    half_mass_radius = np.sort(R)[half_mass_idx]
    if verbosity > 0:
        print('total stellar mass', np.sum(mass))
        print('half stellar mass', half_mass)
        print(np.sort(R)[half_mass_idx], ' half (stellar) mass radius')
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1.plot(np.sort(R), cumulative_mass)
        ax1.axvline(np.sort(R)[half_mass_idx])    
        ax1.axhline(half_mass)
        ax1.axvline(np.sort(R)[half_mass_idx])
        ax1.set_xlim(0,100)
        ax2.plot(np.sort(R), cumulative_mass)
        ax2.axvline(np.sort(R)[half_mass_idx])    
        ax2.axhline(half_mass)
        ax2.axvline(np.sort(R)[half_mass_idx])
    return half_mass_radius, half_mass_idx
    
def compute_roundness_metric(snapshot, verbosity=3):
    '''Computes the roundness metric for a given set of particles
    from an input snapshot, assuming pynbody format
    verbosity (int): level of chattiness, 0 is none, anything above is printy print print
    '''
    mass, x, y, z = snapshot['mass'], snapshot['x'], snapshot['y'], snapshot['z']
    sigma_x = np.sqrt(np.sum(mass*(x**2)))/np.sqrt(np.sum(mass))
    sigma_y = np.sqrt(np.sum(mass*(y**2)))/np.sqrt(np.sum(mass))
    sigma_z = np.sqrt(np.sum(mass*(z**2)))/np.sqrt(np.sum(mass))
    #computing the roundness metric
    roundness = sigma_z/(np.sqrt(sigma_x*sigma_y))
    #in genel et al 2015, they say roundness < 0.55 = flat, while roundness > 0.9 = round
    print('roundness < 0.55 = flat, while roundness > 0.9 = round')
    return roundness

def compute_rotation_metric(dat, halfmass_radius):
    #rotation metric from Sales et al 2010, yoinked from mordor paper
    R = halfmass_radius*3
    top = np.sum(dat['mass'][dat['r']<R]*\
                 (dat['jz'][dat['r']<R]/dat['rxy'][dat['r']<R])**2)
    bottom = np.sum(dat['mass'][dat['r']<R]*np.sum(dat['vel'][dat['r']<R]**2, axis=1))
    return top/bottom

#using disky, bulge subpops to make estimate of disk scale length, scale height, hernquist scale length, mass fraction
#do these look fine? should I drop the prefactors (like .25/(pi a^2) and just assume that rho covers it?)
def hernquist_bulge_log(R, herna, rho0):
    #hernquist bulge profile, takes in spherical R, 
    #hernquist a parameter, rho0 parameter, returns log(density)
    return np.log10(herna**4/(2.0*np.pi*R)*(R+herna)**(-3.0)) + rho0

def exponential_disk_log(R, a, rho0):
    #exponential disk density profile, takes in cylindrical R
    #scale length and rho0
    return np.log10(0.25/(np.pi*a*a) * np.exp(-R/a)) + rho0

def disk_z_log(z, h, rho0):
    #sech^2 vertical disk density profile, takes in z's and scale height, rho0
    sh = 1.0/np.cosh(z/h)
    return np.log10(0.25/(np.pi*h) * sh * sh) + rho0

def disk_bulge_R_log(rr, a, h, herna, Mfac, rho0):
    #combined disk + bulge profile, takes in spherical r
    #scale radius, scale height, hernquist scale parameter, disk mass fraction
    #and rho0, returns density 
    '''Disk + hernquist bulge density profile
    rr array (N): galactocentric spherical R coords
    a (float): Disk scale radius
    h (float): Disk scale height
    herna (float): Hernquist scale length
    Mfac (float): Mass fraction in disk
    '''
    x = np.sqrt(rr/3)
    y = np.sqrt(rr/3)
    z = np.sqrt(rr/3)
    R = np.sqrt(x**2 + y**2)
    sh = 1.0/np.cosh(z/h)
    d1 = 0.25*Mfac/(np.pi*a*a*h) * np.exp(-R/a) * sh * sh
    d2 = (1-Mfac)*herna**4/(2.0*np.pi*rr)*(rr+herna)**(-3.0)
    return np.log10(d1 + d2) + rho0


def density_residual(theta, R_array, real_density):
    '''feed in array of spherical radius coords, corresponding to the bins for the density estimation.
    along with the real density in these bins. Theta here is the parameters for the disk_bulge_xyz function, 
    i.e. a, h, herna, Mfac  -> scale length, scale height, hernquist scale length, mass fraction in disk'''
    predicted_density = disk_bulge_R_log(R_array, *theta)
    residual = predicted_density - real_density
    print(np.sqrt(np.sum(residual**2)))
    return np.sqrt(np.sum(residual**2))

def j_vcirc(dat):
    big_G = 4.3*10**(-6) #kpc km^2 s^-2 Msol^-1
    #total E = - G M / 2 r, vcirc = sqrt(G M / r) -> L = r vcirc = r sqrt(G Menc / r) 
    # r = - G Menc / 2 E -> L = G menc / (sqrt(-2E))
    #potential = G Menc / r
    # Menc = grav_potential * r / G (mass enclosed includes dark matter!)
    menc = dat['phi']*np.sqrt(dat['x']**2+dat['y']**2+dat['z']**2) / big_G
    #this is sort of a weird hybrid, getting mass enclosed from the grav potential 
    # and then usinf the cylindrical radius in the Vcirc
    vcirc = np.sqrt(big_G * menc / np.sqrt(dat['x']**2+dat['y']**2))
    L = vcirc * np.sqrt(dat['x']**2+dat['y']**2)
    return L

def j_circ_func(dat, bins=25, smoothing_size = 3, verbose=3):
    #fitting a curve to the max Lz in each energy bin, following lead of lane and bovy 2024
    #smoothing the curve to get rid of wonky bumps
    lz_max, ebins, num = stats.binned_statistic(dat['phi']+dat['ke'], dat['jz'], statistic='max', bins=bins)
    energies = dat['phi']+dat['ke']
    smoothed_lz_max = uniform_filter1d(lz_max, size=smoothing_size, mode='nearest') #smoothing this out
    j_circs = np.interp(energies, ebins[:-1]+(ebins[1]-ebins[0])/2, smoothed_lz_max)
    if verbose > 0:
        plt.figure()
        plt.scatter(energies, dat['jz'], s=.1, c='k')
        plt.scatter(ebins[:-1]+(ebins[1]-ebins[0])/2, smoothed_lz_max, c='r')
        plt.scatter(energies, j_circs, c='b', s=.1)
        plt.xlabel('Energy')
        plt.ylabel('Jzcirc')
    return j_circs
