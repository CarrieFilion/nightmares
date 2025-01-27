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
from analysis_functions import *
# %%
snap_path = '/mnt/home/dreams/ceph/Sims/CDM/MW_zooms/SB5'
group_path = '/mnt/home/dreams/ceph/FOF_Subfind/CDM/MW_zooms/SB5/'
param_path = '/mnt/home/dreams/ceph/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
print_plots = False
make_basis = True
#%%
#ok box 0 seems to be in box_0/box_0 ....?
#box = 420    #previously did 999 thru the beef
#529 is BEAUTIFUL, 420 is too
#which simulation is used [0,1023] ---- 999 is beautiful boxy-peanut bulge! + spiral!
#23 is suuuper warped and ends up being round label, 28 is a bit warped, 32 is basically all bulge and tiny disk
#91 and 15 is no disk. 5 is nice, 672 is supre nice n disky. 537 is big bulge. 99 wack
snap = 90     #which snapshot 90 -> z=0; will work up to z~1 as written
part_type = 4 #which particle type to calculate the density
#553, 555 don't have enough mass for MW-mass galaxies
#interestingly, it seems that a= 0.01, h = 0.001 or 0.002, and herna = 0.002 or 0.004 seem to
#be p good fits for a lot of the galaxies. a grid probably would work pretty well
    #%%
##by_eye_class = 'disky'

#%%
'''
with open('DREAMS_check_roundness.txt', 'a') as f:
    f.write('galaxy in box ')
    f.write(str(box))
    f.write(' is '),
    f.write(galaxy_type_round)
    f.write(' by roundness metric')
    f.write(' and ')
    f.write(galaxy_type_rot)
    f.write(' by rotation metric')
    f.write(' and by eye: ')
    f.write(by_eye_class)
    f.write("\n")
'''
#%%
#######################################################
# should write in checks to ensure that curve fit, least squares values didnt just converge to edge of bounds
# and then rerun with different method if htey did
#%%
with open('DREAMS_CDM_galaxy_stats.txt', 'w') as f:
    f.write('box_number')
    f.write(' , ')
    f.write('classification')
    f.write(' , ')
    f.write('roundness_metric')
    f.write(' , ')
    f.write('rotation_metric')
    f.write(' , ')
    f.write('jcirc_frac')
    f.write(' , ')
    f.write('scale_length')
    f.write(' , ')
    f.write('scale_height')
    f.write(' , ')
    f.write('hernquist_a')
    f.write(' , ')
    f.write('Mfac')
    f.write(' , ')
    f.write('Mvir')
    f.write(' , ')
    f.write('Rvir')
    f.write("\n")
#%%
from random import randint
boxes = [409, 589, 963, 272, 110, 211, 349, 20, 435, 864, 238, 436, 433, 12, 346]
#[randint(0,1023) for i in range(10)]
#%%
boxes
#%%
#read in our snapshot
for box in boxes:
    dat, extra = load_zoom_particle_data_pynbody(snap_path, group_path, box,
                                                snap, part_type, verbose=3)
    if dat:
        pynbody.analysis.center(dat, mode='pot')
        dat.physical_units()
        half_mass_radius, hm_idx = half_mass(dat, verbosity = 3)
        if half_mass_radius > 5:
            pynbody.analysis.faceon(dat, disk_size = half_mass_radius) #rotate to centered, face-on
        else:   
            pynbody.analysis.faceon(dat, disk_size = 8)
    #compute jcirc from maximum Lz of the data -
    jcirc = j_circ_func(dat, bins=25, smoothing_size = 3)
    #some nice plots to inspect
    disky = (np.sqrt(dat['x']**2 + dat['y']**2) < 30)&(np.abs(dat['z']) < 5) # just r, z cut
    if print_plots == True:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        plt.suptitle('box '+str(box))
        ax1.hist2d(dat['x'][disky], dat['y'][disky], bins=100, norm=LogNorm(), cmap='twilight')
        ax1.set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='y (kpc)')
        ax2.hist2d(dat['x'][disky], dat['z'][disky], bins=100, norm=LogNorm(), cmap='twilight')
        ax2.set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='z (kpc)')

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1.scatter(dat['x'][disky], dat['y'][disky], s=.01, c='k')
        ax1.set(xlim=(-20,20), ylim=(-20,20), xlabel='x (kpc)', ylabel='y (kpc)')
        ax2.scatter(dat['x'][disky], dat['z'][disky], s=.01, c='k')
        ax2.set(xlim=(-20,20), ylim=(-10,10), xlabel='x (kpc)', ylabel='z (kpc)')

        plt.figure()
        plt.title('box '+str(box))
        plt.hist(dat['vphi'][disky], bins=np.arange(-500,500, 10), color='k', histtype='step')
        plt.hist(dat['vphi'][(np.sqrt(dat['x']**2 + dat['y']**2) < 30)&(np.abs(dat['z']) < 5)&
                            (np.sqrt(dat['x']**2 + dat['y']**2) > 3)], bins=np.arange(-500,500, 10),
                            color='grey', ls=':', histtype='step') #dropping wack bulge, high z
        plt.xlabel('vphi (km/s)')

        plt.figure()    
        plt.title('box '+str(box))
        plt.hist(dat['jz']/jcirc, bins=100)
        plt.xlabel('jz/jcirc')
    #make a mask based on circularity and location - cut out centermost bit
    bulge_mask = (dat['jz']/jcirc<0.1)&(dat['r']<5)
    disk_mask = (dat['jz']/jcirc>0.6)&(dat['x']**2+dat['y']**2<30**2)&(np.abs(dat['z'])<5)&(dat['rxy']>2)
    if print_plots == True:
        fig, (ax1, ax2) = plt.subplots(2,2, figsize=(12,12))
        plt.suptitle('box '+str(box))
        ax1[0].scatter(dat['x'][bulge_mask], dat['y'][bulge_mask], s=.01, c='purple')
        ax1[0].set(xlabel='x (kpc)', ylabel='y (kpc)', title='Bulge')
        ax1[1].scatter(dat['x'][bulge_mask], dat['z'][bulge_mask], s=.01, c='purple')
        ax1[1].set(xlabel='x (kpc)', ylabel='z (kpc)', title='Bulge')
        ax2[0].scatter(dat['x'][disk_mask], dat['y'][disk_mask], s=.01, c='k')
        ax2[0].set(xlabel='x (kpc)', ylabel='y (kpc)', title='Disk')
        ax2[1].scatter(dat['x'][disk_mask], dat['z'][disk_mask], s=.01, c='k')
        ax2[1].set(xlabel='x (kpc)', ylabel='z (kpc)', title='Disk')
    #check if it is disky
    roundness = compute_roundness_metric(dat)
    rot = compute_rotation_metric(dat, half_mass_radius)
    jcirc_metric = len(dat[(dat['jz']/jcirc>0.6)&(dat['r']<30)])/len(dat[(dat['r']<30)])
    print(roundness)
    if roundness < 0.55:
        galaxy_type_round = 'disky'
    else:
        galaxy_type_round = 'not disky'
    print('rotation metric, 0.4 and higher have high fraction of mass that rotates around GC', 
        rot)
    if rot > 0.4:
        galaxy_type_rot = 'disky'
    else:
        galaxy_type_rot = 'not disky'
    print('galaxy in box ', box, ' is ', galaxy_type_round, ' by roundness metric', ' and ',
        galaxy_type_rot, ' by rotation metric')
    print('jcirc_metric', jcirc_metric)

    if rot > 0.4 and roundness < 0.55 and jcirc_metric > 0.15:
        galaxy_class = 'disk'
    elif rot > 0.4 and jcirc_metric > 0.15:
        galaxy_class = 'disky?'
    else:
        galaxy_class = 'other'

    if galaxy_class == 'disk' or galaxy_class == 'disky?':
        #fit the disk in r, z and hernquist in R
        ring_density_smoo, rcens = compute_r_density_exp(dat[disk_mask])
        z_density_smoo, zcens = compute_z_density_sinh(dat[disk_mask])
        shell_density_smoo, rrcens = compute_R_density_hern(dat[bulge_mask])

        rexp, pcov = curve_fit(exponential_disk_log, rcens, 
                                ring_density_smoo, bounds=([0.1, 0.1], [15, 500]), maxfev=10000000)
        zexp, pcov = curve_fit(disk_z_log, zcens, z_density_smoo, bounds=([0.01, 0.1],[5,500]), maxfev=10000000)
        bulge, pcov = curve_fit(hernquist_bulge_log, rrcens, shell_density_smoo, bounds=([0,.1],[5,500]), maxfev=10000000)

        print('scale length, scale height, hernquist a', rexp[0], zexp[0], bulge[0])  
        print('rhos', rexp[1], zexp[1], bulge[1])                       
        print('mass ratio, disk/total', np.sum(dat['mass'][disk_mask])/np.sum(dat['mass']))
        print('mass ratio, bulge/total', np.sum(dat['mass'][bulge_mask])/np.sum(dat['mass']))
        mr = np.sum(dat['mass'][disk_mask])/\
                        (np.sum(dat['mass'][disk_mask])+np.sum(dat['mass'][bulge_mask]))
        print('mass ratio, disk/(disk+bulge)', mr)

        #now doing the full R density, using the disk and bulge fits
        mass_rr, rrbins = compute_R_density_hern(dat[disky])
        param_start = np.array([np.round(rexp[0],2), np.round(zexp[0],2), 
                                np.round(bulge[0],2),
                            np.round(mr,2), np.round((rexp[1]+zexp[1])/2,2)])
        print('starting with these estimates:', param_start)
        sol = least_squares(fun = density_residual, x0 = param_start, args=(rrcens, shell_density_smoo),
                            bounds = (np.array([0.1, 0.01, 0, 0.1, .1]), np.array([15, 5, 5, 1, 500])), loss='cauchy')
        print('find scale length, scale height, hernquist a, mfrac, rho0', sol['x'])
        if print_plots == True:
            #lets look at fits -
            fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))
            plt.suptitle('box '+str(box))
            ax1.scatter(rcens, (ring_density_smoo), label='Data')
            ax1.plot(rcens, exponential_disk_log(rcens, rexp[0], rexp[1]), 
                    label='Fit, scale len '+str(round(rexp[0],2)) + ' rho0 '+str(round(rexp[1],2)), c='r')
            ax1.set(yscale='linear', xlabel='R (kpc)', ylabel='log(surface density)')
            ax1.legend()
            ax2.scatter(zcens, z_density_smoo, label='Data')
            ax2.plot(zcens, disk_z_log(zcens, zexp[0], zexp[1]), c='r', 
                    label='Fit scale len '+ str(round(zexp[0], 2)) + ' rho0 '+str(round(zexp[1],2)))
            ax2.set(yscale='linear', xlabel='z (kpc)', ylabel='log(density)')
            ax2.legend()
            ax3.scatter(rrcens, shell_density_smoo, label='Data')
            ax3.plot(rrcens, hernquist_bulge_log(rrcens, bulge[0], bulge[1]), 
                    label='Fit herna'+str(round(bulge[0],2)) + ' rho ' + str(round(bulge[1],2)), c='r')
            ax3.set(yscale='linear', xlabel='r (kpc)', ylabel='log(density)')
            ax3.legend()
        
            fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
            plt.suptitle('box '+str(box))
            ax1.scatter(rrcens, (shell_density_smoo), label='Data')
            ax1.plot(rrcens, disk_bulge_R_log(rrcens, sol['x'][0], sol['x'][1], 
                                                sol['x'][2], sol['x'][3], sol['x'][4]), label='Fit', c='k')
            ax1.set(xlabel='R (kpc)', ylabel='log density')
            ax1.legend()

            fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
            plt.suptitle('box '+str(box))
            ax1.scatter(rrcens, 10**(shell_density_smoo), label='Data')
            ax1.plot(rrcens, 10**disk_bulge_R_log(rrcens, sol['x'][0], sol['x'][1], 
                                                    sol['x'][2], sol['x'][3], sol['x'][4]), 
                    label='Fit '+str(sol['x']), c='k')
            ax1.set(xlabel='R (kpc)', ylabel=' density')
            ax1.legend()
        print('final scale length, scale height, hernquist a, mfrac, rho0', np.around(sol['x'],2))
        #load in the dm halo! then we get the virial mass, virial radius and can scale the disk
        halo, extra = load_zoom_particle_data_pynbody(snap_path, group_path, box,
                                                    snap, 1, verbose=3)
        pynbody.analysis.center(halo, mode='pot')
        halo.physical_units()
        rvir = round(pynbody.analysis.halo.virial_radius(halo).item(),2)
        mvir = round(halo['mass'][(halo['r']<= rvir)].sum().item(),2)
        print('virial radius:', rvir)
        print('virial mass (halo only):', mvir/1e12, ' * 1e12')
        #pynbody.plot.profile.rotation_curve(halo)
        p = pynbody.analysis.profile.Profile(halo, min=0.1, max=100)
        if print_plots == True:
            plt.figure()
            plt.title('box '+str(box))
            plt.plot(p['rbins'], p['v_circ'])
            plt.xlabel('R')
            plt.ylabel('vcirc')
        #getting
        a = round(sol['x'][0]/rvir, 3)
        h = round(sol['x'][1]/rvir, 3)
        hernquist = round(sol['x'][2]/rvir,3)
        Mfac = round(sol['x'][3], 2)
        print(a, h, hernquist, Mfac, ' a, h, hernquist, Mfac in virial radii')

        with open('DREAMS_CDM_galaxy_stats.txt', 'a') as f:
            f.write('box_'+str(box))
            f.write(' , ')
            f.write(galaxy_class)
            f.write(' , ')
            f.write(str(roundness))
            f.write(' , ')
            f.write(str(rot))
            f.write(' , ')
            f.write(str(jcirc_metric))
            f.write(' , ')
            f.write(str(a))
            f.write(' , ')
            f.write(str(h))
            f.write(' , ')
            f.write(str(hernquist))
            f.write(' , ')
            f.write(str(Mfac))
            f.write(' , ')
            f.write(str(mvir))
            f.write(' , ')
            f.write(str(rvir))
            f.write("\n")
            
        if make_basis == True:
            disk_basis_config = f"""
                            id         : cylinder
                            parameters:
                                acyl: {a}       # The scale length of the exponential disk
                                hcyl: {h}       # The scale height of the exponential disk
                                HERNA: {hernquist}       # The scale length of the Hernquist disk
                                Mfac: {Mfac}       # The mass fraction in disk
                                dtype: "diskbulge"   # diskbulge dtype
                                lmaxfid: 72      # The maximum spherical harmonic order for the input basis
                                nmaxfid: 64      # The radial order for the input spherical basis
                                mmax: 8          # The maximum azimuthal order for the cylindrical basis
                                nmax: 18         # The maximum radial order of the cylindrical basis
                                ncylnx: 256      # The number of grid points in mapped cylindrical radius
                                ncylny: 128      # The number of grid points in mapped verical scale
                                ncylodd: 4       # The number of anti-symmetric radial basis functions per azimuthal order m
                                rnum: 500       # The number of radial integration knots in the inner product
                                pnum: 1          # The number of azimuthal integration knots (pnum: 0, assume axisymmetric target density)
                                tnum: 80         # The number of colatitute integration knots
                                ashift: 0.0      # Target shift length in scale lengths to create more variance
                                vflag: 16        # Verbosity flag: print diagnostics to stdout for vflag>0
                                logr: false      # Log scaling in cylindrical radius
                                cachename : disk_dreams_a{a}_h{h}_herna{hernquist}_f{Mfac}.cache      # The cache file name
                            """
            disk_basis = pyEXP.basis.Basis.factory(disk_basis_config)
            xmin = a/5
            xmax = a*10
            numr = 300
            zmin = -h*10
            zmax = h*10
            numz = 300
            symbasis = disk_basis.getBasis(xmin, xmax, numr, zmin, zmax, numz)

            ortho_disk = disk_basis.orthoCheck()
            # Make a table of worst orthgonal checks per harmonic order
  
            form = '{:>4s}  {:>13s}'
            with open('DREAMS_ortho_disk_test_a{}_h{}_herna{}_f{}.txt'.format(a, h, hernquist, Mfac), 'w') as f:
                f.write("Disk ortho check")
                f.write(form.format('l', 'error'))
                f.write(form.format('-', '-----'))
                for l in range(len(ortho_disk)):
                    mat = ortho_disk[l]
                    worst = 0.0
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            if i==j: test = np.abs(1.0 - mat[i, j])
                            else:    test = np.abs(mat[i, j])
                            if test > worst: worst = test
                    f.write('{:4d}  {:13.6e}'.format(l, worst))

            R = np.linspace(xmin, xmax, numr)
            Z = np.linspace(zmin, zmax, numz)

            xv, yv = np.meshgrid(R, Z)
            if print_plots == True:
                fig, (ax) = plt.subplots(8, 18, figsize=(5*18, 5*8))
                for m in range(8):
                    for n in range(18):
                        # Tranpose for contourf
                        ax[m][n].contourf(xv, yv, symbasis[m][n]['potential'].transpose())
                        ax[m][n].set_xlabel('R')
                        ax[m][n].set_ylabel('Z')
                        ax[m][n].set_title('m, n={}, {}'.format(m, n))
                        #fig.colorbar(cx, ax=ax[m][n])
            empirical = disk_basis.createFromArray(dat['mass'][disky]/mvir, 
                                                dat['pos'][disky]/rvir, 
                                                time=0.0)
            empirical_coefs = pyEXP.coefs.Coefs.makecoefs(empirical, 'disk')
            empirical_coefs.add(empirical)
            coefs = empirical_coefs.getAllCoefs()
            power = empirical_coefs.Power()
            for m in range(power.shape[1]):
                print('power in m =', m)
                print(power[0,m])
            print('top 3 ms: ', np.arange(9)[np.argsort(power[0,:])[::-1]][0:3])

            np.save('DREAMS_coefs_box_{}.npy'.format(box), coefs)
            np.save('DREAMS_power_box_{}.npy'.format(box), power)
    else:
        print('fitting hernquist only')
        #fit the disk in r, z and hernquist in R
        shell_density_smoo, rrcens = compute_R_density_hern(dat[(dat['r']<30)])

        roundy, pcov = curve_fit(hernquist_bulge_log, rrcens, shell_density_smoo, maxfev=10000000)

        print('hernquist a', roundy[0])  

        #now doing the full R density, using the disk and bulge fits
        mass_rr, rrbins = compute_R_density_hern(dat[(dat['r']<30)])
        ###### to here ->>>>>
        param_start = np.array([np.round(roundy[0],2), np.round(roundy[1],2)])
        print('starting with these estimates:', param_start)
        sol = least_squares(fun = density_residual_nodisk, x0 = param_start, args=(rrcens, shell_density_smoo),
                            bounds = (np.array([0.1, 0.1]), np.array([30, 500])), loss='cauchy')
        print('find scale length, scale height, hernquist a, mfrac, rho0', sol['x'])
        if print_plots == True:
            #lets look at fits -
            fig, (ax1) = plt.subplots(1,1,figsize=(5,5))
            plt.suptitle('box '+str(box))
            ax1.scatter(rrcens, shell_density_smoo, label='Data')
            ax1.plot(rrcens, hernquist_bulge_log(rrcens, roundy[0], roundy[1]), 
                    label='Fit herna'+str(round(roundy[0],2)) + ' rho ' + str(round(roundy[1],2)), c='r')
            ax1.set(yscale='linear', xlabel='r (kpc)', ylabel='log(density)')
            ax1.legend()
        
            fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
            plt.suptitle('box '+str(box))
            ax1.scatter(rrcens, (shell_density_smoo), label='Data')
            ax1.plot(rrcens, hernquist_bulge_log(rrcens, sol['x'][0], sol['x'][1]), label='Fit', c='k')
            ax1.set(xlabel='R (kpc)', ylabel='log density')
            ax1.legend()

            fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
            plt.suptitle('box '+str(box))
            ax1.scatter(rrcens, 10**(shell_density_smoo), label='Data')
            ax1.plot(rrcens, 10**hernquist_bulge_log(rrcens, sol['x'][0], sol['x'][1]), label='Fit ' + str(sol['x']), c='k')
            ax1.set(xlabel='R (kpc)', ylabel=' density')
            ax1.legend()
        halo, extra = load_zoom_particle_data_pynbody(snap_path, group_path, box,
                                                    snap, 1, verbose=3)
        pynbody.analysis.center(halo, mode='pot')
        halo.physical_units()
        rvir = round(pynbody.analysis.halo.virial_radius(halo).item(),2)
        mvir = round(halo['mass'][(halo['r']<= rvir)].sum().item(),2)
        print('virial radius:', rvir)
        print('virial mass (halo only):', mvir/1e12, ' * 1e12')
        #pynbody.plot.profile.rotation_curve(halo)
        p = pynbody.analysis.profile.Profile(halo, min=0.1, max=100)
        if print_plots == True:
            plt.figure()
            plt.title('box '+str(box))
            plt.plot(p['rbins'], p['v_circ'])
            plt.xlabel('R')
            plt.ylabel('vcirc')
        #getting
        a = 0.0
        h = 0.0
        hernquist = round(sol['x'][0]/rvir,3)
        Mfac = 0.0
        print(a, h, hernquist, Mfac, ' a, h, hernquist, Mfac in virial radii')

        with open('DREAMS_CDM_galaxy_stats.txt', 'a') as f:
            f.write('box_'+str(box))
            f.write(' , ')
            f.write(galaxy_class)
            f.write(' , ')
            f.write(str(roundness))
            f.write(' , ')
            f.write(str(rot))
            f.write(' , ')
            f.write(str(jcirc_metric))
            f.write(' , ')
            f.write(str(a))
            f.write(' , ')
            f.write(str(h))
            f.write(' , ')
            f.write(str(hernquist))
            f.write(' , ')
            f.write(str(Mfac))
            f.write(' , ')
            f.write(str(mvir))
            f.write(' , ')
            f.write(str(rvir))
            f.write("\n")
            
        if make_basis == True:
            disk_basis_config = f"""
                            id         : cylinder
                            parameters:
                                acyl: {a}       # The scale length of the exponential disk
                                hcyl: {h}       # The scale height of the exponential disk
                                HERNA: {hernquist}       # The scale length of the Hernquist disk
                                Mfac: {Mfac}       # The mass fraction in disk
                                dtype: "diskbulge"   # diskbulge dtype
                                lmaxfid: 72      # The maximum spherical harmonic order for the input basis
                                nmaxfid: 64      # The radial order for the input spherical basis
                                mmax: 8          # The maximum azimuthal order for the cylindrical basis
                                nmax: 18         # The maximum radial order of the cylindrical basis
                                ncylnx: 256      # The number of grid points in mapped cylindrical radius
                                ncylny: 128      # The number of grid points in mapped verical scale
                                ncylodd: 4       # The number of anti-symmetric radial basis functions per azimuthal order m
                                rnum: 500       # The number of radial integration knots in the inner product
                                pnum: 1          # The number of azimuthal integration knots (pnum: 0, assume axisymmetric target density)
                                tnum: 80         # The number of colatitute integration knots
                                ashift: 0.0      # Target shift length in scale lengths to create more variance
                                vflag: 16        # Verbosity flag: print diagnostics to stdout for vflag>0
                                logr: false      # Log scaling in cylindrical radius
                                cachename : disk_dreams_a{a}_h{h}_herna{hernquist}_f{Mfac}.cache      # The cache file name
                            """
            disk_basis = pyEXP.basis.Basis.factory(disk_basis_config)
            xmin = a/5
            xmax = a*10
            numr = 300
            zmin = -h*10
            zmax = h*10
            numz = 300
            symbasis = disk_basis.getBasis(xmin, xmax, numr, zmin, zmax, numz)

            ortho_disk = disk_basis.orthoCheck()
            # Make a table of worst orthgonal checks per harmonic order                    
            form = '{:>4s}  {:>13s}'
            with open('DREAMS_ortho_disk_test_a{}_h{}_herna{}_f{}.txt'.format(a, h, hernquist, Mfac), 'w') as f:
                f.write("Disk ortho check")
                f.write(form.format('l', 'error'))
                f.write(form.format('-', '-----'))
                for l in range(len(ortho_disk)):
                    mat = ortho_disk[l]
                    worst = 0.0
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            if i==j: test = np.abs(1.0 - mat[i, j])
                            else:    test = np.abs(mat[i, j])
                            if test > worst: worst = test
                    f.write('{:4d}  {:13.6e}'.format(l, worst))

            R = np.linspace(xmin, xmax, numr)
            Z = np.linspace(zmin, zmax, numz)

            xv, yv = np.meshgrid(R, Z)
            if print_plots == True:
                fig, (ax) = plt.subplots(8, 18, figsize=(5*18, 5*8))
                for m in range(8):
                    for n in range(18):
                        # Tranpose for contourf
                        ax[m][n].contourf(xv, yv, symbasis[m][n]['potential'].transpose())
                        ax[m][n].set_xlabel('R')
                        ax[m][n].set_ylabel('Z')
                        ax[m][n].set_title('m, n={}, {}'.format(m, n))
                        plt.savefig('box'+str(box)+'basis.jpeg')
                        #fig.colorbar(cx, ax=ax[m][n])
            empirical = disk_basis.createFromArray(dat['mass'][(dat['r']<30)]/mvir, 
                                                dat['pos'][(dat['r']<30)]/rvir, 
                                                time=0.0)
            empirical_coefs = pyEXP.coefs.Coefs.makecoefs(empirical, 'disk')
            empirical_coefs.add(empirical)
            coefs = empirical_coefs.getAllCoefs()
            power = empirical_coefs.Power()
            for m in range(power.shape[1]):
                print('power in m =', m)
                print(power[0,m])
            print('top 3 ms: ', np.arange(9)[np.argsort(power[0,:])[::-1]][0:3])

            np.save('DREAMS_coefs_box_{}.npy'.format(box), coefs)
            np.save('DREAMS_power_box_{}.npy'.format(box), power)

#%%
#round(halo['mass'][(halo['r']<= rvir)].sum(),2)
                #%%
#halo['mass'][(halo['r']<= rvir)].sum().item()
#halo['mass'].sum()
halo, extra = load_zoom_particle_data_pynbody(snap_path, group_path, 409,
                                                    snap, 1, verbose=3)
halo['mass'].sum()
#%%
boxes
#%%
'''
vmin = np.minimum(np.min(np.real(coefs.T)[0,:,:]), np.min(np.imag(coefs.T)[0,:,:]))
vmax = np.maximum(np.max(np.real(coefs.T)[0,:,:]), np.max(np.imag(coefs.T)[0,:,:]))
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,10))
cb = ax1.imshow(np.real(coefs.T)[0,:,:], cmap='twilight', vmin=vmin, vmax=vmax, origin='lower')
ax1.set(xlabel='m', ylabel='n', title='Real Coefs', xticks=range(9), yticks=range(18))
fig.colorbar(cb, ax=ax1, fraction=.08)
cb = ax2.imshow(np.imag(coefs.T)[0,:,:], cmap='twilight',vmin=vmin, vmax=vmax, origin='lower')
ax2.set(xlabel='m', ylabel='n', title='Imaginary Coefs', xticks=range(9), yticks=range(18))
fig.colorbar(cb, ax=ax2, fraction=0.08)
cb = ax3.imshow(np.sqrt(np.imag(coefs.T)[0,:,:]**2 + np.real(coefs.T)[0,:,:]**2), 
                cmap='magma', origin='lower')
ax3.set(xlabel='m', ylabel='n', title='sqrt(imaginary^2 + real^2) Coefs', 
        xticks=range(9), yticks=range(18))
fig.colorbar(cb, ax=ax3, fraction=0.08)
plt.subplots_adjust(wspace=0.5)
plt.savefig('coefs.jpeg')
#%%
pmin  = [-xmax, -xmax, -zmax]
pmax  = [xmax, xmax, -zmax]
grid  = [  150,   150,   150]
fields = pyEXP.field.FieldGenerator([0.0], pmin, pmax, grid)
empirical_surfaces = fields.slices(disk_basis, empirical_coefs)
#%%
fig, ax = plt.subplots(1,2, figsize=(12,6))
cb1 = ax[0].hist2d(dat['pos'][disky][:,0], dat['pos'][disky][:,1], 
                   bins=200, norm=LogNorm(), cmap='magma')
ax[0].set(xlim=(-35, 35),ylim=(-35, 35), xlabel='x (kpc)', ylabel='y (kpc)')
cb3 = ax[1].imshow(empirical_surfaces[0.0]['dens'].T,  cmap='magma', 
             origin='lower', extent=[-xmax, xmax, -xmax, xmax],norm=LogNorm(vmin=1,
                                                vmax=np.max(cb1[0].flatten())))
ax[1].set(xlabel='x (rvir)', ylabel='y (rvir)',xlim=(-.1, .1), ylim=(-.1, .1))
fig.colorbar(cb1[3], ax=ax[0])
fig.colorbar(cb3, ax=ax[1])
plt.savefig('density_slice_xy.jpeg')
#%%
empirical_volume = np.zeros((150, 150, 200))
z_sice = np.linspace(-zmax, zmax, 200)

for i in range(len(z_sice)):
    pmin  = [-xmax, -xmax, z_sice[i]]
    pmax  = [xmax, xmax, z_sice[i]]
    grid  = [  150,   150,   0]
    fields = pyEXP.field.FieldGenerator([0.0], pmin, pmax, grid)
    empirical_surfaces = fields.slices(disk_basis, empirical_coefs)
    empirical_volume[:,:,i] = empirical_surfaces[0.0]['dens']

cmap = plt.get_cmap('magma').copy()
#cmap.set_under('black')
#cmap.set_bad('black')
fig, ax = plt.subplots(1,2, figsize=(12,2))
cb1 = ax[0].hist2d(dat['pos'][disky][:,0], dat['pos'][disky][:,2], 
                   bins=200, norm=LogNorm(), cmap=cmap)
ax[0].set(xlim=(-35, 35),ylim=(-6, 6), xlabel='x (kpc)', ylabel='z (kpc)')
cb3 = ax[1].imshow(np.sum(empirical_volume, axis=1).T,  cmap=cmap, 
             origin='lower', extent=[-xmax, xmax, -zmax, zmax],
             norm=LogNorm(vmin=1,vmax=np.max(cb1[0].flatten())))
#cb3.cmap.set_under('k')
#cb3.cmap.set_bad('k')
ax[1].set(xlabel='x (rvir)', ylabel='z (rvir)',xlim=(-.1, .1))
fig.colorbar(cb1[3], ax=ax[0])
fig.colorbar(cb3, ax=ax[1])
#%%
cmap = plt.get_cmap('magma').copy()
cmap.set_under('black')
cmap.set_bad('black')
fig, ax = plt.subplots(1,2, figsize=(13,6))
cb1 = ax[0].hist2d(dat['pos'][disky][:,0], dat['pos'][disky][:,1], 
                   bins=200, norm=LogNorm(), cmap=cmap)
ax[0].set(xlim=(-35, 35),ylim=(-35, 35), xlabel='x (kpc)', ylabel='y (kpc)')
cb3 = ax[1].imshow(np.sum(empirical_volume, axis=2).T,  cmap=cmap, 
             origin='lower', extent=[-xmax, xmax, -xmax, xmax],
             norm=LogNorm())
#cb3.cmap.set_under('k')
#cb3.cmap.set_bad('k')
ax[1].set(xlabel='x (rvir)', ylabel='y (rvir)',xlim=(-.1, .1), ylim=(-.1,.1))
fig.colorbar(cb1[3], ax=ax[0])
fig.colorbar(cb3, ax=ax[1])
#%%
'''
#np.sum(empirical_volume, axis=1).shape
#%%
'''empirical_volume = np.zeros((150, 150, 200))
z_sice = np.linspace(-zmax, zmax, 200)

for i in range(len(z_sice)):
    pmin  = [-xmax, -xmax, z_sice[i]]
    pmax  = [xmax, xmax, z_sice[i]]
    grid  = [  150,   150,   0]
    fields = pyEXP.field.FieldGenerator([0.0], pmin, pmax, grid)
    empirical_surfaces = fields.slices(disk_basis, empirical_coefs)
    empirical_volume[:,:,i] = empirical_surfaces[0.0]['dens']

fig, ax = plt.subplots(1,2, figsize=(12,6))
cb1 = ax[0].hist2d(dat['pos'][disky][:,0], dat['pos'][disky][:,1], 
                   bins=200, norm=LogNorm(), cmap='magma')
ax[0].set(xlim=(-35, 35),ylim=(-35, 35), xlabel='x (kpc)', ylabel='y (kpc)')
cb3 = ax[1].imshow((np.mean(empirical_volume, axis=2)).T,  cmap='magma', 
             origin='lower', extent=[-xmax, xmax, -xmax, xmax],norm=LogNorm(vmin=1,
                                                vmax=np.max(cb1[0].flatten())))
ax[1].set(xlabel='x (rvir)', ylabel='y (rvir)',xlim=(-.1, .1), ylim=(-.1, .1))
fig.colorbar(cb1[3], ax=ax[0])
fig.colorbar(cb3, ax=ax[1])

fig, ax = plt.subplots(1,2, figsize=(12,2))
cb1 = ax[0].hist2d(dat['pos'][disky][:,0], dat['pos'][disky][:,2], 
                   bins=200, norm=LogNorm(), cmap='magma')
ax[0].set(xlim=(-35, 35),ylim=(-6, 6), xlabel='x (kpc)', ylabel='z (kpc)')
cb3 = ax[1].imshow(np.mean(empirical_volume, axis=1).T,  cmap='magma', 
             origin='lower', extent=[-xmax, xmax, -zmax, zmax],
             norm=LogNorm(vmin=1,vmax=np.max(cb1[0].flatten())))
ax[1].set(xlabel='x (rvir)', ylabel='z (rvir)',xlim=(-.1, .1), ylim=(-zmax, zmax))
fig.colorbar(cb1[3], ax=ax[0])
fig.colorbar(cb3, ax=ax[1])
# %%
plt.imshow(empirical_volume[:,10,:].T,extent=[-xmax, xmax, -zmax, zmax],
           cmap='magma', norm=LogNorm(vmin=1,vmax=np.max(cb1[0].flatten())))
plt.ylim(-zmax, zmax)
#plt.xlim(-.1, .1)
empirical_volume.shape'''
# %%
h*rvir, zmax*rvir
# %%
