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
    this is original from Jonah
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
def get_MW_idx(cat, model):
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
    if model == 'CDM':
        mcut = (tot_masses > 5e11) & (tot_masses < 2.5e12)
    elif model == 'WDM':
        mcut = (tot_masses > 7e11) & (tot_masses < 2.5e12)
    else:
        print('no galaxies with this model yet!')
    if True in np.unique(mcut):
        contamination = masses[:,2] / tot_masses
        idx = np.argmin(contamination[mcut])
        mw_idx = np.arange(len(masses))[mcut][idx]
    else:
        mw_idx = None
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
    to remove subhaloes, set sub_idx to 0
    """
    
    if fof_idx < 0 and sub_idx < 0:
        return part_cat, group_cat
    
    if fof_idx < 0 and sub_idx >= 0:
        fof_idx = group_cat['SubhaloGrNr'][sub_idx]
    
    offsets = np.sum(group_cat['GroupLenType'][:fof_idx],axis=0)
    sub_start = group_cat['GroupFirstSub'][fof_idx] #index of central halo in group
    if sub_idx >= 1:
        #specify one specific subhalo
        sub_loc = sub_start + sub_idx #sub start is index of central, sub_idx is number subhalo in the group
        nsubs = 1
        offsets += np.sum(group_cat['SubhaloLenType'][sub_start:sub_loc], axis=0)
        num_parts = group_cat['SubhaloLenType'][sub_loc]
    elif sub_idx < 0:
        #all particles in group, including those not assigned subhalos
        num_parts = group_cat['GroupLenType'][fof_idx]
        nsubs = group_cat['GroupNsubs'][fof_idx]
    else:
        #sub_idx == 0 -> only particles in central subhalo of whatever group is specified, 
        #this effectively removes subhaloes + non-assigned stars from central
        num_parts = group_cat['SubhaloLenType'][sub_start] 
        nsubs = 1
    
    
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
            if sub_idx >= 1:
                new_group_cat[key] = group_cat[key][sub_loc:sub_loc+nsubs]
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
    grp_cat = load_group_data(fof_path, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 
                                         'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr',
                                         'SubhaloHalfmassRadType', 'SubhaloHalfmassRad'])
    model = group_path.split('/')[-4]
    mw_idx = get_MW_idx(grp_cat, model) 
    if mw_idx is None:
        print('No haloes in MW mass range! Returning empty catalogs')
        return
    else:
        prt_cat, fof_cat = get_galaxy_data(part_cat, grp_cat, mw_idx)
        return prt_cat, fof_cat
    return 

def load_particle_data_alt(path, part_types):
    """
    revised from original, will use all keys instead of passing in subset 
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
            if pt <= 5:
                keys = ofile[f'PartType{pt}'].keys()
                for key in keys:
                    if pt == 1 and key == 'Masses':
                        cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                    else:
                        if f'PartType{pt}/{key}' in ofile:
                            cat[f'PartType{pt}/{key}'] = np.array(ofile[f'PartType{pt}/{key}'])
            else:
                print('Particle type does not exist, try an integer <= 5')
                return
    return cat

def load_zoom_particle_data_pynbody(snap_path, group_path, box, snap, part_type,  subhaloes = False, verbosity=1):
    '''take in the snapshot path, the group path, the number box that you want
    the snapshot of, the snapshot number (i.e. what time, here z ~ 0 = 90), 
    the particle type. This will load all keys and port the data into pynbody with the correct cosmology
                    0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
    pass in whether or not you want to load subhaloes (no subhaloes = False, default)
    '''
    #load in to find scale factor:
    path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
    check = h5py.File(path)
    a = 1/(check['Header'].attrs['Redshift']+1)
    check.close()
    #read in the IC file that has the cosmological parameters, update the particle reader with right cosmology
    #while the below is loading in CDM parameters, there is this info in WDM,  so I think no harm in doing this each time
    #need to reconsider once ADM etc are run
    param_info = np.loadtxt(f'{snap_path}/box_{box}/aux_files/ics_config.txt',  dtype='str', skiprows=20, max_rows=6)
    param_dic = {}
    for i in range(len(param_info)):
        param_dic[param_info[i][0]] = param_info[i][2]
    
    pynbody.config['omegaM0'] = float(param_dic['Omega_m'])
    pynbody.config['omegaL0'] = float(param_dic['Omega_L'])
    pynbody.config['h'] = float(param_dic['H0'])/100 #should be .6909, but file gives 69.09
    pynbody.config['omegaB0'] = float(param_dic['Omega_b'])
    pynbody.config['sigma8'] = float(param_dic['sigma_8'])
    pynbody.config['ns'] = float(param_dic['nspec'])
    pynbody.config['a'] = a
    pynbody.units.a = a
    pynbody.units.h = float(param_dic['H0'])/100

    print('loading in ')

    fof_path = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
    grp_cat = load_group_data(fof_path, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', \
                                            'GroupPos', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloPos',
                                            'SubhaloHalfmassRadType', 'SubhaloHalfmassRad'])
    model = group_path.split('/')[-4]
    
    name_map = pynbody.snapshot.namemapper.AdaptiveNameMapper('gadgethdf-name-mapping',return_all_format_names=False)
    if snap == 90:
        print('snap is z = 0 -> loading in the MW-mass halo')
        mw_idx = get_MW_idx(grp_cat, model) 
        if mw_idx is None:
            print('No MW-like mass systems in this box!')
            return None, None 
        offsets = np.sum(grp_cat['GroupLenType'][:mw_idx],axis=0)
        sub_start = grp_cat['GroupFirstSub'][mw_idx] #new edit
        if subhaloes == False:
            print('removing subhaloes')
            num_parts = grp_cat['SubhaloLenType'][grp_cat['GroupFirstSub'][mw_idx]] #want sub_idx = 0 in Jonah's code
            nsubs = 1
        else:
            print('keeping all subhaloes')
            num_parts = grp_cat['GroupLenType'][mw_idx]
            nsubs = grp_cat['GroupNsubs'][mw_idx]

    
    if snap != 90:
        print('snap is z > 0 -> not positive this works!')
        #get MW-mass halo from z = 0
        print('reading in z = 0 first to get halo...')
        fof_path90 = f'{group_path}/box_{box}/fof_subhalo_tab_{90:03}.hdf5'
        grp_cat90 = load_group_data(fof_path90, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', \
                                         'GroupPos', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloPos',
                                         'SubhaloHalfmassRadType', 'SubhaloHalfmassRad'])
        model = group_path.split('/')[-4]
        mw_idx = get_MW_idx(grp_cat90, model) 
        #can't use the z = 0 MW mass idx finder here - need to identify the MW-mass halo at z = 0
        #and then trace it thru time. Will need to load the merger tree to identify the correct halo
        tree_cat = h5py.File(group_path+'/box_'+str(int(box))+'/tree_extended.hdf5')
        nnodes = len(tree_cat['SubhaloID'])
        #initialize the search by finding the target halo in the tree
        target_cut = np.isin(tree_cat['SubhaloGrNr'], mw_idx) & np.isin(tree_cat['SnapNum'], 90)
        target  = tree_cat['FirstSubhaloInFOFGroupID'][target_cut][0]
        target_idx = tree_cat['SubhaloID'] == target
        start = tree_cat['MainLeafProgenitorID'][target_idx]
        start_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == start][0]
        start_snap = tree_cat['SnapNum'][start_idx]
        next_ID = tree_cat['DescendantID'][start_idx]
        next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0] 
        grp_start_offset_over_time = dict()
        grp_idx_over_time = dict()
        grp_start_offset_over_time[start_snap] = np.sum(grp_cat90['GroupLenType'][:mw_idx], axis=0)
        grp_idx_over_time[start_snap] = mw_idx
        if verbosity > 0:
            print('start snap', start_snap)
            print('zero start offset', grp_start_offset_over_time[start_snap])
        #loop through to identify the location of the first subhalo in the group,
        #this is the massive central galaxy that we want to use
        group_first_sub_over_time = dict()
        group_first_sub_over_time[start_snap] = tree_cat['GroupFirstSub'][start_idx]
        grp_len_type_over_time = dict()
        grp_len_type_over_time[start_snap] = tree_cat['GroupLenType'][start_idx]
        grp_nsubs_over_time = dict()
        grp_nsubs_over_time[start_snap] = tree_cat['GroupNsubs'][start_idx]
        while next_ID != -1:
            this_snap = tree_cat['SnapNum'][next_idx]
            next_ID = tree_cat['DescendantID'][next_idx]
            haloiddxxx = tree_cat['SubhaloGrNr'][next_idx]
            fof_path2 = f'{group_path}/box_{box}/fof_subhalo_tab_{this_snap:03}.hdf5'
            grp_cat2 = load_group_data(fof_path2, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', \
                                        'GroupPos', 'SubhaloLenType', 'SubhaloGrNr', 'SubhaloPos',
                                        'SubhaloHalfmassRadType', 'SubhaloHalfmassRad'])
            grp_idx_over_time[this_snap] = haloiddxxx
            grp_start_offset_over_time[this_snap] = np.sum(grp_cat2['GroupLenType'][:haloiddxxx], axis=0)
            group_first_sub_over_time[this_snap] = tree_cat['GroupFirstSub'][next_idx]
            grp_len_type_over_time[this_snap] = tree_cat['GroupLenType'][next_idx]
            grp_nsubs_over_time[this_snap] = tree_cat['GroupNsubs'][next_idx]
            if next_ID == -1:
                break
            next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]
            if verbosity > 0:
                print('this snap', this_snap)
                print('group number', haloiddxxx)
                print('grp len type over time', grp_len_type_over_time[this_snap])
                print('comp to this snap:', grp_cat2['GroupLenType'][haloiddxxx])


        offsets = grp_start_offset_over_time[snap]
        sub_start = group_first_sub_over_time[snap] #new edit
        if subhaloes == False:
            print('removing subhaloes')
            num_parts = grp_cat['SubhaloLenType'][group_first_sub_over_time[snap]] #want sub_idx = 0 in Jonah's code
            nsubs = 1
            mw_idx = grp_idx_over_time[snap]
        else:
            print('keeping all subhaloes')
            num_parts = grp_len_type_over_time[snap]
            nsubs = grp_nsubs_over_time[snap]
            mw_idx = grp_idx_over_time[snap]
        
    pt = int(part_type)

    soft = np.loadtxt(f'{snap_path}/box_{box}/aux_files/parameters-usedvalues',  dtype='str')
    soft_dict = {}
    if pt <= 5:
        for i in range(len(soft)):
            soft_dict[soft[i][0]] = soft[i][1]
        comoving = float(soft_dict['SofteningComovingType'+str(pt)])
        maxphys = float(soft_dict['SofteningMaxPhysType'+str(pt)])
        #####a = 1.0 #bad to hardcode this in, but pynbody complains if I set a in the config
        if comoving > maxphys/a:
            comoving = maxphys/a
        
    if verbosity > 0:
        print('offsets', offsets)
        print('sub_start', sub_start)
        print('nsubs', nsubs)
        print('num_parts', num_parts)
        print('mw_idx, sub_start', mw_idx, sub_start)

    dat = load_particle_data_alt(path, pt)
    if dat is None:
        return None, None 
    new_group_cat = dict()
    for key in grp_cat:
        if 'Group' in key:
            new_group_cat[key] = grp_cat[key][sub_start]
        else:
            new_group_cat[key] = grp_cat[key][sub_start:sub_start+nsubs]
    if verbosity > 0:
        print('-----------------')
        print('newgroupcat grouppos, subhalopos', new_group_cat['GroupPos'], new_group_cat['SubhaloPos'])
        print('-----------------')
        print('recentering on zeroeth subhalo, position: ', new_group_cat['SubhaloPos'][0,:])

    dat[f'PartType{pt}/Coordinates'] = dat[f'PartType{pt}/Coordinates'] - new_group_cat['SubhaloPos'][0,:]
    unit_dict = {'BirthPos': units.kpc * units.h**-1,
            'BirthVel': units.a**1/2 * units.km * units.s**-1,
            'Coordinates': units.kpc *units.a * units.h**-1,
            'GFM_InitialMass': 1e10 * units.Msol * units.h**-1,
            'GFM_Metallicity': units.Unit(1),
            'GFM_Metals': units.Unit(1),
            'GFM_MetalsTagged': units.Unit(1),
            'GFM_StellarFormationTime': units.Unit(1),
            'GFM_StellarPhotometrics': units.Unit(1),
            'Masses': 1e10 * units.Msol * units.h**-1,
            'ParticleIDs': units.Unit(1),
            'Potential': units.km**2 * units.s**-2 * units.a**-1,
            'SubfindDMDensity': 1e10 * units.Msol * units.h**2 * units.kpc**-3*units.a**-3,
            'SubfindDensity': 1e10 * units.Msol * units.h**2 * units.kpc**-3*units.a**-3,
            'SubfindHsml': units.kpc * units.a * units.h**-1,
            'SubfindVelDisp': units.km * units.s**-1,
            'Velocities': units.km * units.a**1/2 * units.s**-1,
            'CenterOfMass': units.kpc * units.a * units.h**-1,
            'Density': 1e10 * units.Msol * units.h**2 * units.kpc**-3*units.a**-3,
            'ElectronAbundance': units.Unit(1),
            'GFM_AGNRadiation': units.erg * units.s**-1 * units.cm**-2 * 4 * np.pi,
            'GFM_CoolingRate': units.erg * units.s**-1 * units.cm**3,
            'GFM_Metallicity': units.Unit(1),
            'GFM_Metals': units.Unit(1),
            'GFM_MetalsTagged': units.Unit(1),
            'GFM_WindDMVelDisp': units.km * units.s**-1,
            'GFM_WindHostHaloMass': 1e10 * units.Msol * units.h**-1,
            'InternalEnergy': units.km**2 * units.s**-2,
            'MagneticField': units.h * units.a**-2 * 1e5 * units.Msol**1/2 * units.kpc**-1/2* units.km * units.s**-1 * units.kpc**-1,
            'MagneticFieldDivergence': units.h**3 * 1e5 * units.Msol**1/2 * units.km * units.a**-2 * units.s**-1 * units.kpc**-5/2 * units.a**-5/2,
            'NeutralHydrogenAbundance': units.Unit(1),
            'StarFormationRate': units.Msol * units.yr**-1,
            'InternalEnergy': units.km**2 * units.s**-2,
            'AllowRefinement': units.Unit(1),
            'HighResGasMass': units.Unit(1)} #high res gas mass isn't defined in the tng webpage, gonna just set unit to one
    if part_type == 0:
        print('loading gas particles for snapshot ', snap, 
        ' of box ', box)
        gas = pynbody.new(gas=len(dat[f'PartType{pt}/Masses'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        #not picked up from config...
        gas.properties['omegaM0'] = float(param_dic['Omega_m'])
        gas.properties['omegaL0'] = float(param_dic['Omega_L'])
        gas.properties['h'] = float(param_dic['H0'])/100 #should be .6909, but file gives 69.09
        gas.properties['omegaB0'] = float(param_dic['Omega_b'])
        gas.properties['sigma8'] = float(param_dic['sigma_8'])
        gas.properties['ns'] = float(param_dic['nspec'])
        gas.properties['a'] = float(a)
        gas['eps'] = pynbody.array.SimArray([comoving], 'kpc', dtype='float64')
        for key in dat.keys():
            key = key.split('/')[1]
            mapped_name = name_map(key, reverse=True)
            gas[mapped_name] = dat[f'PartType{pt}/{key}'][offsets[pt]:offsets[pt]+num_parts[pt]]
            gas[mapped_name].units = unit_dict[key]
        return gas, new_group_cat
    elif part_type == 1:
        print('loading high res dark matter particles for snapshot ', snap, 
        ' of box ', box)
        dm = pynbody.new(dm = len(dat[f'PartType{pt}/ParticleIDs'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        dm.properties['omegaM0'] = float(param_dic['Omega_m'])
        dm.properties['omegaL0'] = float(param_dic['Omega_L'])
        dm.properties['h'] = float(param_dic['H0'])/100 #should be .6909, but file gives 69.09
        dm.properties['omegaB0'] = float(param_dic['Omega_b'])
        dm.properties['sigma8'] = float(param_dic['sigma_8'])
        dm.properties['ns'] = float(param_dic['nspec'])
        dm.properties['a'] = float(a)
        dm['eps'] = pynbody.array.SimArray([comoving], 'kpc', dtype='float64')
        with h5py.File(path) as ofile:
            dm['Masses'] = np.ones(ofile['PartType1/ParticleIDs'][offsets[pt]:offsets[pt]+num_parts[pt]].shape)*ofile['Header'].attrs['MassTable'][1]
            mapped_name = name_map('Masses', reverse=True)
            dm[mapped_name].units = unit_dict['Masses']
            dm[mapped_name] = np.ones(ofile['PartType1/ParticleIDs'][offsets[pt]:offsets[pt]+num_parts[pt]].shape)*ofile['Header'].attrs['MassTable'][1]
        for key in dat.keys():
            key = key.split('/')[1]
            #dm[key] = dat[f'PartType{pt}/{key}'][offsets[pt]:offsets[pt]+num_parts[pt]]
            mapped_name = name_map(key, reverse=True)
            dm[mapped_name] = dat[f'PartType{pt}/{key}'][offsets[pt]:offsets[pt]+num_parts[pt]]
            dm[mapped_name].units = unit_dict[key]
        return dm, new_group_cat
    elif part_type == 2:
        print('loading low res dark matter particles for snapshot ', snap, 'of box ', box)
        print('are you sure that you want this component?')
        dm = pynbody.new(dm = len(dat[f'PartType{pt}/ParticleIDs'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        dm.properties['omegaM0'] = float(param_dic['Omega_m'])
        dm.properties['omegaL0'] = float(param_dic['Omega_L'])
        dm.properties['h'] = float(param_dic['H0'])/100 #should be .6909, but file gives 69.09
        dm.properties['omegaB0'] = float(param_dic['Omega_b'])
        dm.properties['sigma8'] = float(param_dic['sigma_8'])
        dm.properties['ns'] = float(param_dic['nspec'])
        dm.properties['a'] = float(a)
        dm['eps'] = pynbody.array.SimArray([comoving], 'kpc', dtype='float64')
        for key in dat.keys():
            key = key.split('/')[1]
            #dm[key] = dat[f'PartType{pt}/{key}'][offsets[pt]:offsets[pt]+num_parts[pt]]
            mapped_name = name_map(key, reverse=True)
            dm[mapped_name] = dat[f'PartType{pt}/{key}'][offsets[pt]:offsets[pt]+num_parts[pt]]
            dm[mapped_name].units = unit_dict[key]
        return dm, new_group_cat
    elif part_type == 3:
        print(' tracers particles for snapshot ', snap, 'of box ', box, ' NOT USED IN DREAMS')
    elif part_type == 4:
        print('loading star particles for snapshot ', snap, 'of box ', box)
        star = pynbody.new(star=len(dat[f'PartType{pt}/Masses'][offsets[pt]:offsets[pt]+num_parts[pt]]))
        star.properties['omegaM0'] = float(param_dic['Omega_m'])
        star.properties['omegaL0'] = float(param_dic['Omega_L'])
        star.properties['h'] = float(param_dic['H0'])/100 #should be .6909, but file gives 69.09
        star.properties['omegaB0'] = float(param_dic['Omega_b'])
        star.properties['sigma8'] = float(param_dic['sigma_8'])
        star.properties['ns'] = float(param_dic['nspec'])
        star.properties['a'] = float(a)
        star['eps'] = pynbody.array.SimArray([comoving], 'kpc', dtype='float64')
        for key in dat.keys():
            key = key.split('/')[1]
            mapped_name = name_map(key, reverse=True)
            star[mapped_name] = dat[f'PartType{pt}/{key}'][offsets[pt]:offsets[pt]+num_parts[pt]]
            star[mapped_name].units = unit_dict[key]
        return star, new_group_cat
    elif part_type == 5:
        print('loading black hole particles for snapshot ', snap, 'of box ', box)
        print('we cant do black holes yet, sorry!') #we /could/ do black holes, just need
        #to reconfigure pynbody to recognize them as a particle type
        #blackhole = pynbody.new(bh = len(dat.bh[key][offsets[pt]:offsets[pt]+num_parts[pt]]))
        #for key in dat.blackhole.loadable_keys():
        #    blackhole[key] = dat.blackhole[key][offsets[pt]:offsets[pt]+num_parts[pt]]
        #    if blackhole[key].units == 1.00e+00 or blackhole[key].units == units.NoUnit():
        #        blackhole[key].units = units.Unit(1)
        #    return blackhole, new_group_cat
    else:
        print('invalid particle type')
        return None, None 

    return 
# %%
