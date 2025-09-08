#%%
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import h5py, os
from scipy.spatial import KDTree
from pynbody import units
import pynbody
from reader_funcs import *
from matplotlib.colors import LogNorm

# %%
#%%
snap_path = '/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5/'
group_path = '/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/'
param_path = '/mnt/home/dreams/ceph/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
# %%
box = 34       #which simulation is used [0,1023]
snap = 90     #which snapshot 90 -> z=0; will work up to z~1 as written
part_type = 4 #which particle type to calculate the density
#%%
#read in the sim snapshot, here we remove subhaloes
dat, grp_dat = load_zoom_particle_data_pynbody(snap_path, group_path, box, snap, part_type, subhaloes=False)
# %%
#%%
#center the galaxy, convert to physical units, and rotate to face-on
pynbody.analysis.center(dat, mode='ssc')
dat.physical_units()
pynbody.analysis.faceon(dat, disk_size = 5) 
#%%
#just plot a density histogram to check it out
fig, ax = plt.subplot_mosaic([['A','B']], figsize=(10,5))
ax['A'].hexbin(dat['x'], dat['y'], norm=LogNorm(), cmap='twilight', gridsize=800)
ax['A'].set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='y (kpc)')
ax['B'].hexbin(dat['x'], dat['z'], bins=100, norm=LogNorm(), cmap='twilight', gridsize=800)
ax['B'].set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='z (kpc)')
# %%
