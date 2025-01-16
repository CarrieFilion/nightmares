#%%#%%
import sys, os
os.chdir('/mnt/home/cfilion/DREAMS/')
#! pwd
#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py, os
from scipy.spatial import KDTree
from reader_funcs import *
import pynbody
from scipy import stats
from matplotlib.colors import LogNorm
import sys, os
sys.path.append('/mnt/home/cfilion/DREAMS/EXP-tools/')
sys.path.append('/mnt/home/cfilion/DREAMS/EXP/lib/')
sys.path.append('/mnt/home/cfilion/DREAMS/EXP-tools/EXPtools/')
import pyEXP
from basis_builder import makemodel
from inspect import getmembers, isfunction
from reader_funcs import *
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit, least_squares
# %%
snap_path = '/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5'
group_path = '/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/'
param_path = '/mnt/home/dreams/ceph/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
#%%
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
        print('total mass', np.sum(mass))
        print('half mass', half_mass)
        print(np.sort(R)[half_mass_idx], ' half mass radius')
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
    mass (array): mass of particles
    x (array): x positions of particles
    y (array): y positions of particles
    z (array): z positions of particles
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
#do these look fine? should I just drop the prefactors (like .25/(pi a^2) and just assume that rho covers it?)
def hernquist_bulge_log(R, herna, rho0):
    return np.log10(herna**4/(2.0*np.pi*R)*(R+herna)**(-3.0)) + rho0

def exponential_disk_log(R, a, rho0):
    return np.log10(0.25/(np.pi*a*a) * np.exp(-R/a)) + rho0

def disk_z_log(z, h, rho0):
    sh = 1.0/np.cosh(z/h)
    return np.log10(0.25/(np.pi*h) * sh * sh) + rho0

def disk_bulge_xyz_log(rr, a, h, herna, Mfac, rho0):
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
    return np.log10(d1) + np.log10(d2) + rho0;

## could do like z being integrated quantity, s
def density_residual(theta, xyz_array, real_density):
    '''feed in Nx3 array of xyz coords, corresponding to the bins for the density estimation.
    along with the real density in these bins. Theta here is the parameters for the disk_bulge_xyz function, 
    i.e. a, h, herna, Mfac  -> scale length, scale height, hernquist scale length, mass fraction in disk'''
    predicted_density = disk_bulge_xyz_log(xyz_array, *theta)
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

#%%
#%%
#ok box 0 seems to be in box_0/box_0 ....?
box = 999      #which simulation is used [0,1023] ---- 999 is beautiful boxy-peanut bulge! + spiral!
#23 is suuuper warped and ends up being round label, 28 is a bit warped, 32 is basically all bulge and tiny disk
#91 and 15 is no disk. 5 is nice, 672 is supre nice n disky. 537 is big bulge
snap = 90     #which snapshot 90 -> z=0; will work up to z~1 as written
part_type = 4 #which particle type to calculate the density
h = .6909     #reduced hubble constant (do not change)
#%%
#read in our snapshot
dat, extra = load_zoom_particle_data_pynbody(snap_path, group_path, box,
                                              snap, part_type, verbose=3)
pynbody.analysis.center(dat, mode='pot')
dat.physical_units()
half_mass_radius, hm_idx = half_mass(dat, verbosity = 3)
if half_mass_radius > 5:
    pynbody.analysis.faceon(dat, disk_size = half_mass_radius) #rotate to centered, face-on
else:   
    pynbody.analysis.faceon(dat, disk_size = 8)
#%%
#check if it is disky
print(compute_roundness_metric(dat))
print('rotation metric, 0.4 and higher have high fraction of mass that rotates around GC', 
      compute_rotation_metric(dat, half_mass_radius))
print('computing Jcirc')
jcirc = j_circ_func(dat, bins=25, smoothing_size = 3)
#%%
#some nice plots to inspect
disky = (np.sqrt(dat['x']**2 + dat['y']**2) < 30)&(np.abs(dat['z']) < 5) # just r, z cut
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.hist2d(dat['x'][disky], dat['y'][disky], bins=100, norm=LogNorm(), cmap='binary')
ax1.set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='y (kpc)')
ax2.hist2d(dat['x'][disky], dat['z'][disky], bins=100, norm=LogNorm(), cmap='binary')
ax2.set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='z (kpc)')

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.scatter(dat['x'][disky], dat['y'][disky], s=.01, c='k')
ax1.set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='y (kpc)')
ax2.scatter(dat['x'][disky], dat['z'][disky], s=.01, c='k')
ax2.set(xlim=(-20,20), ylim=(-10,10), xlabel='x (kpc)', ylabel='z (kpc)')

plt.figure()
plt.hist(dat['vphi'][disky], bins=np.arange(-500,500, 10), color='k', histtype='step')
plt.hist(dat['vphi'][(np.sqrt(dat['x']**2 + dat['y']**2) < 30)&(np.abs(dat['z']) < 5)&
                     (np.sqrt(dat['x']**2 + dat['y']**2) > 3)], bins=np.arange(-500,500, 10),
                       color='grey', ls=':', histtype='step') #dropping wack bulge, high z
plt.xlabel('vphi (km/s)')
#%%
#make a mask based on circularity and location - cut out centermost bit
bulge_mask = (dat['jz']/jcirc<0.1)&(dat['r']<5)
disk_mask = (dat['jz']/jcirc>0.6)&(dat['x']**2+dat['y']**2<30**2)&(np.abs(dat['z'])<5)&(dat['rxy']>2)
fig, (ax1, ax2) = plt.subplots(2,2, figsize=(12,12))
ax1[0].scatter(dat['x'][bulge_mask], dat['y'][bulge_mask], s=.01, c='purple')
ax1[1].scatter(dat['x'][bulge_mask], dat['z'][bulge_mask], s=.01, c='purple')
ax2[0].scatter(dat['x'][disk_mask], dat['y'][disk_mask], s=.01, c='k')
ax2[1].scatter(dat['x'][disk_mask], dat['z'][disk_mask], s=.01, c='k')
#%%
#fit the disk in r, z and hernquist in R
mass_r, rbins, num = stats.binned_statistic(dat['rxy'][disk_mask],
                                   dat['mass'][disk_mask], statistic='sum', bins=100)
#getting mass in rings
rcens = rbins[:-1]+((rbins[1]-rbins[0])/2)
ring_density = mass_r/(np.pi*rbins[1:]**2-np.pi*rbins[:-1]**2)
ring_density_smoo = np.log10(uniform_filter1d(ring_density, size=10, mode='nearest'))


mass_z, zbins, num = stats.binned_statistic(abs(dat['z'][(disk_mask)&(dat['rxy']<11)&(dat['rxy']>1)]),
                                   dat['mass'][(disk_mask)&(dat['rxy']<11)&(dat['rxy']>1)], 
                                   statistic='sum', bins=100)
#getting mass in rings
zcens = zbins[:-1]+((zbins[1]-zbins[0])/2)
z_density = mass_z/(np.pi*10**2 * (zbins[1:]-zbins[:-1])) #used 10kpc radial section, pi r^2 * height
z_density_smoo = np.log10(uniform_filter1d(z_density, size=10, mode='nearest'))


mass_rr, rrbins, num = stats.binned_statistic(dat['r'][bulge_mask],
                                   dat['mass'][bulge_mask], statistic='sum', bins=100)
#getting mass in shells
rrcens = rrbins[:-1]+((rrbins[1]-rrbins[0])/2)
shell_density = mass_rr/(4/3*np.pi*(rrbins[1:]**3-rrbins[:-1]**3))
shell_density_smoo = np.log10(uniform_filter1d(shell_density, size=10, mode='nearest'))

rexp, pcov = curve_fit(exponential_disk_log, rcens, 
                           (ring_density_smoo), maxfev=10000000)
zexp, pcov = curve_fit(disk_z_log, zcens, z_density_smoo, maxfev=10000000)
bulge, pcov = curve_fit(hernquist_bulge_log, rrcens, shell_density_smoo, maxfev=10000000)
print('scale length, scale height, hernquist a', rexp[0], zexp[0], bulge[0])  
print('rhos', rexp[1], zexp[1], bulge[1])                       
print('mass ratio, disk/total', np.sum(dat['mass'][disk_mask])/np.sum(dat['mass']))
print('mass ratio, bulge/total', np.sum(dat['mass'][bulge_mask])/np.sum(dat['mass']))
mr = np.sum(dat['mass'][disk_mask])/\
                (np.sum(dat['mass'][disk_mask])+np.sum(dat['mass'][bulge_mask]))
print('mass ratio, disk/(disk+bulge)', mr)
#%%
#lets look at fits -
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
ax1.scatter(rcens, (ring_density_smoo), label='Data')
ax1.plot(rcens, exponential_disk_log(rcens, rexp[0], rexp[1]), 
         label='Fit, scale len '+str(round(rexp[0],2)) + ' rho0 '+str(round(rexp[1],2)), c='r')
ax1.set(yscale='linear', xlabel='R (kpc)', ylabel='M(r)/A(r) (Msol kpc^-2)')
ax1.legend()
ax2.scatter(zcens, z_density_smoo, label='Data')
ax2.plot(zcens, disk_z_log(zcens, zexp[0], zexp[1]), c='r', 
         label='Fit scale len '+ str(round(zexp[0], 2)) + ' rho0 '+str(round(zexp[1],2)))
ax2.set(yscale='linear', xlabel='z (kpc)', ylabel='M(r)/A(r) (Msol kpc^-2)')
ax2.legend()
ax3.scatter(rrcens, shell_density_smoo, label='Data')
ax3.plot(rrcens, hernquist_bulge_log(rrcens, bulge[0], bulge[1]), 
         label='Fit herna'+str(round(bulge[0],2)) + ' rho ' + str(round(bulge[1],2)), c='r')
ax3.set(yscale='linear', xlabel='r (kpc)', ylabel='M(r)/A(r) (Msol kpc^-2)')
ax3.legend()
#%%
#now doing the full R density, using the disk and bulge fits
mass_rr, rrbins, num = stats.binned_statistic(dat['r'][disky],
                                   dat['mass'][disky], statistic='sum', bins=100)
#getting mass in rings
rrcens = rrbins[:-1]+((rrbins[1]-rrbins[0])/2)
shell_density = mass_rr/(4/3*np.pi*(rrbins[1:]**3-rrbins[:-1]**3))
shell_density_smoo = np.log10(uniform_filter1d(shell_density, size=10, mode='nearest'))

param_start = np.array([np.round(rexp[0],2), np.round(zexp[0],2), np.round(bulge[0],2),
                      np.round(mr,2), np.round((rexp[1]+zexp[1])/2,2)])
print('starting with these estimates:', param_start)
sol = least_squares(fun = density_residual, x0 = param_start, args=(rrcens, shell_density_smoo),
                    bounds = (np.array([0.1, 0.01, 0, 0.2, .1]), np.array([15, 5, 5, 1, 500])), loss='cauchy')
print('find scale length, scale height, hernquist a, mfrac, rho0', sol['x'])
# %%
fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
ax1.scatter(rrcens, (shell_density_smoo), label='Data')
ax1.plot(rrcens, disk_bulge_xyz_log(rrcens, sol['x'][0], sol['x'][1], 
                                    sol['x'][2], sol['x'][3], sol['x'][4]), label='Fit', c='k')
ax1.set(xlabel='R (kpc)', ylabel='log density')
ax1.legend()

fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
ax1.scatter(rrcens, 10**(shell_density_smoo), label='Data')
ax1.plot(rrcens, 10**disk_bulge_xyz_log(rrcens, sol['x'][0], sol['x'][1], 
                                        sol['x'][2], sol['x'][3], sol['x'][4]), label='Fit', c='k')
ax1.set(xlabel='R (kpc)', ylabel=' density')
ax1.legend()
# %%
print('final scale length, scale height, hernquist a, mfrac, rho0', np.around(sol['x'],2))
# %%
