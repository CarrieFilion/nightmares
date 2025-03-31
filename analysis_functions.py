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
    # almost exactly matches halfmass from the group catalog
    mass = snapshot['mass']
    R = snapshot['r']
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
    if verbosity > 0:
        print('roundness < 0.55 = flat, while roundness > 0.9 = round')
        print('computed roundness:', roundness)
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
    #np.log(herna**4/(2.0*np.pi*R)*(R+herna)**(-3.0)) + rho0
    return 4*np.log(herna) - np.log(R) - 3 * np.log(herna + R) + rho0 -1.83788

def exponential_disk_log(r, a, rho0):
    #exponential disk density profile, takes in cylindrical r
    #scale length and rho0
    #np.log(0.25/(np.pi*a*a) * np.exp(-r/a)) + rho0
    return -2.53102 + rho0 - r/a - 2 * np.log(a)

def disk_z_log(z, h, rho0):
    #sech^2 vertical disk density profile, takes in z's and scale height, rho0
    sh = 1.0/np.cosh(z/h)
    #np.log(0.25/(np.pi*h) * sh * sh) + rho0
    return -2.53102 - np.log(h) + 2*np.log(sh) + rho0


def disk_bulge_rz_log(r_bins,z_bins, a, h, herna, Mfac, rho0):
    #combined disk + bulge profile, takes in clyndrical r, z
    #scale radius, scale height, hernquist scale parameter, disk mass fraction
    #and rho0, returns density 
    '''Disk + hernquist bulge density profile
    r_bins array (N): galactocentric cylindrical R coords
    z_bins array (N): galactocentric z coords
    a (float): Disk scale radius
    h (float): Disk scale height
    herna (float): Hernquist scale length
    Mfac (float): Mass fraction in disk
    '''
    rr = np.sqrt(r_bins**2 + z_bins**2)
    sh = 1.0/np.cosh(z_bins/h)
    d1 = 0.25*Mfac/(np.pi*a*a*h) * np.exp(-r_bins/a) * sh * sh
    d2 = (1-Mfac)*herna**4/(2.0*np.pi*rr)*(rr+herna)**(-3.0)
    return np.log(d1 + d2) + rho0

def compute_r_density_exp(dat, binsize=0.5, smooth=True):
    #compute the 2d surface density as a function of cylindrical radius
    #pass in your snapshot (ideally with the proper selection of disk stars),
    #number of desired bins, and whether or not you want smoothing
    #returns density in rings and the ring centers
    mass_r, rbins, num = stats.binned_statistic(dat['rxy'],
                                    dat['mass'], statistic='sum', 
                                    bins=np.arange(dat['rxy'].min(), dat['rxy'].max()+binsize, binsize))
    #getting mass in rings
    rcens = rbins[:-1]+((rbins[1]-rbins[0])/2)
    ring_density = mass_r/(np.pi*rbins[1:]**2-np.pi*rbins[:-1]**2)
    if smooth == True:
        ring_density = uniform_filter1d(ring_density, size=5, mode='nearest')
    ring_density[(ring_density == 0.0)] = 1e-3
    return np.log(ring_density), rcens

def compute_R_density_hern(dat, binsize=0.5, smooth=True):
    #compute 3d density as a function of spherical R
    #pass in your snapshot (ideally with the proper selection of disk stars),
    #number of desired bins, and whether or not you want smoothing
    #returns density in shell and the shell centers
    mass_rr, rrbins, num = stats.binned_statistic(dat['r'],
                                   dat['mass'], statistic='sum', 
                                   bins=np.arange(0,dat['r'].max()+binsize, binsize))
    #getting mass in shells
    rrcens = rrbins[:-1]+((rrbins[1]-rrbins[0])/2)
    shell_density = mass_rr/(4/3*np.pi*(rrbins[1:]**3-rrbins[:-1]**3))
    if smooth == True:
        shell_density = uniform_filter1d(shell_density, size=5, mode='nearest')
    shell_density[(shell_density == 0.0)] = 1e-3
    return np.log(shell_density), rrcens

def compute_z_density_sinh(dat, binsize=.5, smooth=True, outer = 12, inner = 2):
    #compute density as a function of z, within inner < cylindrical r < outer
    #pass in your snapshot (ideally with the proper selection of disk stars),
    #number of desired bins, and whether or not you want smoothing
    #returns density in slices and the slice centers
    mass_z, zbins, num = stats.binned_statistic((dat['z'][(dat['rxy']<outer)&(dat['rxy']>inner)]),
                                    dat['mass'][(dat['rxy']<outer)&(dat['rxy']>inner)], 
                                    statistic='sum', 
                    bins=np.arange(dat['z'].min(), dat['z'].max()+binsize, binsize))
    #getting mass in rings
    zcens = zbins[:-1]+((zbins[1]-zbins[0])/2)
    z_density = mass_z/(np.pi*(outer**2 - inner**2) * (zbins[1:]-zbins[:-1])) 
    if smooth == True:
        z_density = uniform_filter1d(z_density, size=5, mode='nearest')
    z_density[(z_density == 0.0)] = 1e-3 #avoiding log(0) errors
    return np.log(z_density), zcens

def compute_r_z_density(dat, r_binsize=1, z_binsize=0.5):
    #compute density as a function of r, z
    #pass in your snapshot (ideally with the proper selection of disk stars),
    #number of desired bins returns density in slices and the slice centers
    zz_mid = np.array([])
    rr_mid = np.array([])
    mass_rz = np.array([])
    rbins = np.arange(dat['rxy'].min(), dat['rxy'].max()+r_binsize, r_binsize)
    zbins = np.arange(dat['z'].min(), dat['z'].max()+z_binsize, z_binsize)
    for i in range(len(rbins)-1):
        r_low = rbins[i]
        r_high = rbins[i+1]
        if i == len(rbins)-2:
            dat_sel = dat[(dat['rxy'] >= r_low) & (dat['rxy'] <= r_high)]
        else:
            dat_sel = dat[(dat['rxy'] >= r_low) & (dat['rxy'] < r_high)]
        for q in range(len(zbins)-1):
            z_low = zbins[q]
            z_high = zbins[q+1]
            rr_mid = np.append(rr_mid, (r_low+r_high)/2)
            zz_mid = np.append(zz_mid, (z_low+z_high)/2)
            if q == len(zbins)-2:
                dat_rz = dat_sel[(dat_sel['z'] >= z_low) & (dat_sel['z'] <= z_high)]
            else:
                dat_rz = dat_sel[(dat_sel['z'] >= z_low) & (dat_sel['z'] < z_high)]
            #volume of a cylindrical shell = 2 pi r h delta r, r = 1/2(r_out + r_in), delta r = rout - rin
            volume_element = 2 * np.pi * (z_high-z_low) * (r_high - r_low) * (.5*(r_high+r_low))
            mass_rz = np.append(mass_rz, np.sum(dat_rz['mass'])/volume_element)
    mass_rz[(mass_rz == 0.0)] = 1e-3 #avoiding log(0) errors
    return np.log(mass_rz), rr_mid, zz_mid

def density_residual(theta, rr_array, zz_array, real_density):
    '''feed in array of cylindrical radius, z coords, corresponding to the bins for the density estimation.
    along with the real density in these bins. Theta here is the parameters for the disk_bulge_R_log function, 
    i.e. a, h, herna, Mfac  -> scale length, scale height, hernquist scale length, mass fraction in disk'''
    predicted_density = disk_bulge_rz_log(rr_array, zz_array, *theta)
    res = np.abs(np.exp(predicted_density) - np.exp(real_density))
    return np.log(res)


def density_residual_nodisk(theta, R_array, real_density):
    '''feed in array of spherical radius coords, corresponding to the bins for the density estimation.
    along with the real density in these bins. Theta here is the parameters for the hernquist_bulge_log function, 
    i.e. herna, rho0  -> hernquist scale length, density at r=0'''
    predicted_density = hernquist_bulge_log(R_array, *theta)
    res = np.abs(np.exp(predicted_density) - np.exp(real_density))
    return np.log(res)



def j_circ_func(dat, bins=25, smoothing_size = 5, verbosity=3):
    #fitting a curve to the max Lz in each energy bin, following lead of lane and bovy 2024
    #smoothing the curve to get rid of wonky bumps
    lz_max, ebins, num = stats.binned_statistic(dat['phi']+dat['ke'], dat['jz'], statistic='max', bins=bins)
    energies = dat['phi']+dat['ke']
    smoothed_lz_max = uniform_filter1d(lz_max, size=smoothing_size, mode='nearest') #smoothing this out
    j_circs = np.interp(energies, ebins[:-1]+(ebins[1]-ebins[0])/2, smoothed_lz_max)
    if verbosity > 0:
        plt.figure()
        plt.scatter(energies, dat['jz'], s=.1, c='k')
        plt.scatter(ebins[:-1]+(ebins[1]-ebins[0])/2, smoothed_lz_max, c='r')
        plt.scatter(energies, j_circs, c='b', s=.1)
        plt.xlabel('Energy')
        plt.ylabel('Jzcirc')
    return j_circs


def lookup_element(element):
    ''' a lookup function to get the solar abundance of an element and its atomic weight
    solar abundances from Aspland et al 2009 review, in the form log(elem) = log(elem/H) + 12
    atomic weights from CIAAW IUPAC website'''
    if element == 'Mg':
        log_scale = 7.6
        mass = 24.305
    elif element == 'Fe':
        log_scale = 7.5
        mass = 55.847
    elif element == 'H':
        log_scale = 12
        mass = 1.00784
    elif element == 'He':
        log_scale = 10.93
        mass = 4.002
    elif element == 'C':
        log_scale = 8.43
        mass = 12.0096
    elif element == 'N':
        log_scale = 7.83
        mass = 14.00643
    elif element == 'O':
        log_scale = 8.69
        mass = 15.999
    elif element == 'Ne':
        log_scale = 7.93
        mass = 20.1797
    elif element == 'Si':
        log_scale = 7.51
        mass = 28.084
    return log_scale, mass

def compute_z_massrat(element_logabun, element_mass):
    #compute the mass ratio abundance of an element, basing everything off of H
    #assuming solar abundance of H is 12, using the abundance format of the 
    #asplund et al 2009 review, i.e. log(element) = log(element/H) + 12
    #where element and H are in terms of number. Want to convert to mass 
    #using asplund values and mass of H from CIAAW IUPAC website
    logh = 12
    h_mass = 1.00784
    h_massfrac = 0.7154
    return 10**(element_logabun - logh) * element_mass / h_mass * h_massfrac

def abundance_solarscale_fe(element, dat):
    #compute the abundance of an element in terms of solar mass fraction
    #using the solar abundance of Fe as the reference. This is the normal
    # log10(z_element/z_element_sun) - log10(z_fe/z_fe_sun)
    element_log_scale, element_mass = lookup_element(element)
    z_element_sun = compute_z_massrat(element_log_scale, element_mass)
    fe_log_scale, fe_mass = lookup_element('Fe')
    z_fe_sun = compute_z_massrat(fe_log_scale, fe_mass)
    if element == 'He':
        return np.log10(dat['GFM_Metals'][:,1]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    elif element == 'C':
        return np.log10(dat['GFM_Metals'][:,2]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    elif element == 'N':
        return np.log10(dat['GFM_Metals'][:,3]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    elif element == 'O':
        return np.log10(dat['GFM_Metals'][:,4]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    elif element == 'Ne':
        return np.log10(dat['GFM_Metals'][:,5]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    elif element == 'Mg':
        return np.log10(dat['GFM_Metals'][:,6]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    elif element == 'Si':
        return np.log10(dat['GFM_Metals'][:,7]/z_element_sun) - np.log10(dat['GFM_Metals'][:,8]/z_fe_sun)
    return 

def abundance_solarscale_h(element, dat):
    #compute the abundance of an element in terms of solar mass fraction
    #using the solar abundance of Fe as the reference. This is the normal
    # log10(z_element/z_element_sun) - log10(z_fe/z_fe_sun)
    element_log_scale, element_mass = lookup_element(element)
    z_element_sun = compute_z_massrat(element_log_scale, element_mass)
    h_log_scale, h_mass = lookup_element('H')
    z_h_sun = compute_z_massrat(h_log_scale, h_mass)
    if element == 'He':
        return np.log10(dat['GFM_Metals'][:,1]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'C':
        return np.log10(dat['GFM_Metals'][:,2]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'N':
        return np.log10(dat['GFM_Metals'][:,3]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'O':
        return np.log10(dat['GFM_Metals'][:,4]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'Ne':
        return np.log10(dat['GFM_Metals'][:,5]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'Mg':
        return np.log10(dat['GFM_Metals'][:,6]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'Si':
        return np.log10(dat['GFM_Metals'][:,7]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    elif element == 'Fe':
        return np.log10(dat['GFM_Metals'][:,8]/z_element_sun) - np.log10(dat['GFM_Metals'][:,0]/z_h_sun)
    return
# %%
