#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py, os
from scipy.spatial import KDTree
from pynbody import units
import pynbody
# %%
#%%
h = .6909     #reduced hubble constant (do not change)
#%%
def load_particle_data(path, keys, part_types):
    """
    Read particle data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
    """
    cat = dict()
    with h5py.File(path) as ofile:
        
        if type(part_types) == type(0):
            part_types = [part_types]
        
        for pt in part_types:
            for key in keys:
                if pt == 1 and key == 'Masses':
                    cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                else:
                    if f'PartType{pt}/{key}' in ofile:
                        cat[f'PartType{pt}/{key}'] = np.array(ofile[f'PartType{pt}/{key}'])
    return cat
def get_MW_idx(cat):
    """
    Selects the corrent MW-mass galaxy from each simulation given the group catalog.
    This function only works for z~0
    It selects the least contaminated halo with a mass within current uncertainties of the MW's mass
    
    Inputs 
     - cat - a dictionary containing the 'GroupMassType' field from the FOF catalogs
     
    Returns
     - mw_idx - the index into the group catalog for the target MW-mass galaxy
    """
    masses = cat['GroupMassType'] * 1e10 / h
    
    tot_masses = np.sum(masses,axis=1)
    mcut = (tot_masses > 7e11) & (tot_masses < 2.5e12)
    
    contamination = masses[:,2] / tot_masses
    idx = np.argmin(contamination[mcut])
    
    mw_idx = np.arange(len(masses))[mcut][idx]
    return mw_idx
def load_group_data(path, keys):
    """
    Read Group Data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      
    Returns
      cat - a dictionary that contains all of the group and subhalo information for the specified keys
    """
    cat = dict()
    with h5py.File(path) as ofile:
        for key in keys:
            if 'Group' in key:
                cat[key] = np.array(ofile[f'Group/{key}'])
            if 'Subhalo' in key:
                cat[key] = np.array(ofile[f'Subhalo/{key}'])
    return cat

def get_galaxy_data(part_cat, group_cat, fof_idx=-1, sub_idx=-1):
    """
    Given particle and group catalogs, return a new catalog that only contains data for a specified galaxy.
    If fof_idx is given but sub_idx is not, data for the FOF group and all satellites are returned
    If fof_idx is given and sub_idx is given, data for just the specified subhalo of that group
      e.g. fof_idx=3 sub_idx=5 will provide data for the fifth subhalo of group three
    If only sub_idx is supplied, the data for that subfind galaxy is returned, can be in any FOF group
    
    Inputs
      part_cat  - a dictionary containing particle data make from load_particle_data
      group_cat - a dictionary containing group data make from load_particle_data
                  must contain these fields: GroupLenType, GroupFirstSub, GroupNsubs, SubhaloLenType, SubhaloGrNr
      fof_idx   - the FOF group that you want data for
      sub_idx   - the Subfind galaxy that you want data for
      
    Returns
      new_part_cat  - a new dictionary with keys from part_cat but data only for the specified galaxy
      new_group_cat - a new dictionary with keys from group_cat but data only for the specified galaxy
    """
    
    if fof_idx < 0 and sub_idx < 0:
        return part_cat, group_cat
    
    if fof_idx < 0 and sub_idx >= 0:
        fof_idx = group_cat['SubhaloGrNr'][sub_idx]
    
    offsets = np.sum(group_cat['GroupLenType'][:fof_idx],axis=0)
    
    if sub_idx >= 1:
        start_sub = group_cat['GroupFirstSub'][fof_idx]
        offsets += np.sum(group_cat['SubhaloLenType'][start_sub:start_sub+sub_idx], axis=0)
    
    if sub_idx < 0:
        num_parts = group_cat['GroupLenType'][fof_idx]
        nsubs = group_cat['GroupNsubs'][fof_idx]
        sub_start = group_cat['GroupFirstSub'][fof_idx]
    else:
        num_parts = group_cat['SubhaloLenType'][sub_idx]
        nsubs = 1
        sub_start = sub_idx
    
    
    new_part_cat = dict()
    for key in part_cat:
        pt = int(key.split("/")[0][-1])
        print(key, pt)
        new_part_cat[key] = part_cat[key][offsets[pt]:offsets[pt]+num_parts[pt]]
    
    new_group_cat = dict()
    for key in group_cat:
        if 'Group' in key:
            new_group_cat[key] = group_cat[key][fof_idx]
        else:
            new_group_cat[key] = group_cat[key][sub_start:sub_start+nsubs]
    
    return new_part_cat, new_group_cat

def fibonacci_sphere(samples, r):
    """
    This function returns a set of points equally spaced on the surface of a sphere.
    This function was adapted from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    
    Inputs
      samples - the number of points on the sphere
      r       - the radius of the sphere that is sampled
      
    Returns
      points  - the coordinates of the points of the sphere with shape (samples,3)
    """
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / (samples - 1)) * 2  
        radius = np.sqrt(1 - y * y) 

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    points = np.array(points) * r
        
    return points

def calc_density(matter_coords, matter_mass, sample_coords, DesNgb):
    """
    This function takes particle data and calculates the density around a given set of points.
    The points are assumed to be for one radius and are averaged together before returned.
    This function will work for any particle type given to it.
    
    Inputs
      matter_coords - the coordinates for the simulation particles used to calculate the density
      matter_mass   - the mass of the simulation particles, must be the same shape as matter_coords
      sample_coords - the coordinates where the density is calculated
      DesNgb        - the number of simulation particles used to calculate the density (32 is standard)
      
    Returns
      density - the average density for the supplied sample coordinates
    """
    tree = KDTree(matter_coords)
    distance, idx = tree.query(sample_coords, DesNgb)
    hsml = distance[:,-1]
    mass_enclosed = np.sum(matter_mass[idx], axis=1)
    density =  mass_enclosed / (4 / 3 * np.pi * np.power(hsml,3))
    
    density = np.average(density)
    return density

#%%
def load_zoom_particle_data(snap_path, group_path, box, snap, part_type, key_list):
    '''take in the snapshot path, the group path, the number box that you want
    the snapshot of, the snapshot number (i.e. what time, here z ~ 0 = 90), 
    the particle type, and the list of keys you want to load
                       0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
    '''
    print('loading in ')
    if part_type == 0:
        print(' gas particles for snapshot ', snap, 
        ' of box ', box)
    elif part_type == 1:
        print(' high res dark matter particles for snapshot ', snap, 
        ' of box ', box)
    elif part_type == 2:
        print(' low res dark matter particles for snapshot ', snap, 'of box ', box)
    elif part_type == 3:
        print(' tracers particles for snapshot ', snap, 'of box ', box, ' NOT USED IN DREAMS')
    elif part_type == 4:
        print(' star particles for snapshot ', snap, 'of box ', box)
    elif part_type == 5:
        print(' black hole particles for snapshot ', snap, 'of box ', box)
    else:
        print('invalid particle type')
        return None
    path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
    part_cat = load_particle_data(path, key_list, part_type)

    fof_path = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
    grp_cat = load_group_data(fof_path, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr'])

    mw_idx = get_MW_idx(grp_cat) 
    prt_cat, fof_cat = get_galaxy_data(part_cat, grp_cat, mw_idx)
    return prt_cat, fof_cat

def load_zoom_particle_data_pynbody(snap_path, group_path, box, snap, part_type):
    '''take in the snapshot path, the group path, the number box that you want
    the snapshot of, the snapshot number (i.e. what time, here z ~ 0 = 90), 
    the particle type, and the list of keys you want to load
                       0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
    '''
    if snap==90:
        h = .6909
        pynbody.units.h = h
        pynbody.units.a = 1
    else:
        print('not z = 0 - need to revise!')
        return None
    print('loading in ')
    if part_type == 0:
        print(' gas particles for snapshot ', snap, 
        ' of box ', box)
    elif part_type == 1:
        print(' high res dark matter particles for snapshot ', snap, 
        ' of box ', box)
    elif part_type == 2:
        print(' low res dark matter particles for snapshot ', snap, 'of box ', box)
    elif part_type == 3:
        print(' tracers particles for snapshot ', snap, 'of box ', box, ' NOT USED IN DREAMS')
    elif part_type == 4:
        print(' star particles for snapshot ', snap, 'of box ', box)
    elif part_type == 5:
        print(' black hole particles for snapshot ', snap, 'of box ', box)
    else:
        print('invalid particle type')
        return None
    path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
    fof_path = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
    grp_cat = load_group_data(fof_path, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr'])

    mw_idx = get_MW_idx(grp_cat) 


    dat = pynbody.load(path)
    offsets = np.sum(grp_cat['GroupLenType'][:mw_idx],axis=0)
    num_parts = grp_cat['GroupLenType'][mw_idx]
    print(num_parts)
    nsubs = grp_cat['GroupNsubs'][mw_idx]
    sub_start = grp_cat['GroupFirstSub'][mw_idx]
    print(nsubs)
    pt = int(part_type)

    new_group_cat = dict()
    for key in grp_cat:
        if 'Group' in key:
            new_group_cat[key] = grp_cat[key][mw_idx]
        else:
            new_group_cat[key] = grp_cat[key][sub_start:sub_start+nsubs]

    dat['pos'] = dat['pos'] - new_group_cat['GroupPos']

    if part_type == 0:
        gas = pynbody.new(gas=len(dat.gas['pos'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        for key in dat.gas.loadable_keys():
            gas[key] = dat.gas[key][offsets[pt]:offsets[pt]+num_parts[pt]]
            if gas[key].units == 1.00e+00 or gas[key].units == units.NoUnit():
                gas[key].units = units.Unit(1)
        return gas, new_group_cat
    elif part_type == 1:
        dm = pynbody.new(dm = len(dat.dm['pos'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        for key in dat.dm.loadable_keys():
            dm[key] = dat.dm[key][offsets[pt]:offsets[pt]+num_parts[pt]]
            if dm[key].units == 1.00e+00 or dm[key].units == units.NoUnit():
                dm[key].units = units.Unit(1)
        return dm, new_group_cat
    elif part_type == 2:
        dm = pynbody.new(dm = len(dat.dm['pos'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        for key in dat.dm.loadable_keys():
            dm[key] = dat.dm[key][offsets[pt]:offsets[pt]+num_parts[pt]]
            if key == 'Masses':
                with h5py.File(path) as ofile:
                    dm['Masses'] = np.ones(dat.dm['pos'].shape[0])*ofile['Header'].attrs['MassTable'][1]
            if dm[key].units == 1.00e+00 or dm[key].units == units.NoUnit():
                dm[key].units = units.Unit(1)
        return dm, new_group_cat
    elif part_type == 3:
        print(' tracers particles for snapshot ', snap, 'of box ', box, ' NOT USED IN DREAMS')
    elif part_type == 4:
        star = pynbody.new(len(dat.star['pos'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        for key in dat.star.loadable_keys():
            star[key] = dat.star[key][offsets[pt]:offsets[pt]+num_parts[pt]]
            if star[key].units == 1.00e+00 or star[key].units == units.NoUnit():
                star[key].units = units.Unit(1)
        return star, new_group_cat
    elif part_type == 5:
        blackhole = pynbody.new(dat.blackhole[key][offsets[pt]:offsets[pt]+num_parts[pt]])
        for key in dat.blackhole.loadable_keys():
            blackhole[key] = dat.blackhole[key][offsets[pt]:offsets[pt]+num_parts[pt]]
            if blackhole[key].units == 1.00e+00 or blackhole[key].units == units.NoUnit():
                blackhole[key].units = units.Unit(1)
            return blackhole, new_group_cat

    return 