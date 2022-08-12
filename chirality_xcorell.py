#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from ase.io import read as ase_read
from numpy import pi

def rotate(pos, angle):
    """Rotate pos (2D) of angle (in degree)"""
    roto_mtr = np.array([[np.cos(angle/180*np.pi), -np.sin(angle/180*np.pi)],
                         [np.sin(angle/180*np.pi), np.cos(angle/180*np.pi)]])
    pos = np.dot(roto_mtr, pos.T).T # NumPy inverted convention on row/col
    return pos

def cross_correlation2D(o1, o2):
    """Compute cross correlation between matrix o1 and o2

    Use numpy framework. Cross correlation definition here http://paulbourke.net/miscellaneous/correlate/
    """

    o1m, o2m = np.average(o1), np.average(o2)
    o1norm, o2norm = np.sqrt(np.sum((o1-o1m)**2)), np.sqrt(np.sum((o2-o2m)**2))
    return np.sum(((o1-o1m)*(o2-o2m)))/(o1norm*o2norm)

def chirality_xcorellation(pos, pattern, d_vec, gdens=300, plot=False, tth=0, R=1e30):
    """Compute chirality of pattern defined on pos as mininimum of 1-cross correlation across vector of delay angles d_vec

    The pattern (and its x-inversion) are interpolated (linearly) on a grid of gdens(=300 default) points.

    tth is reference angle for plotting purposes, can be ignored.
    R defined a cutoff radius to apply to the signal, in case you want to explude some irregularities.
    """

    # Invert along x axis
    xinv = np.array([[-1,0],[0,1]])
    posxinv = pos@xinv

    # Define regular grid to interpolate
    dens = gdens # How many point to interpolate on?
    x = np.linspace(min(pos[:, 0]), max(pos[:, 0]), dens)
    y = np.linspace(min(pos[:, 1]), max(pos[:, 1]), dens)
    xx, yy = np.meshgrid(x, y)
    q = pattern
    grid_method = 'linear' # 'cubic'# 'linear' # 'nearest' # Choose the best one for your needs.

    # Interpolate ORIGINAL
    gq = griddata((pos[:, 0], pos[:, 1]), pattern, (xx, yy), method=grid_method)

    # Set grid value outside of circle to 0
    rr = np.stack([xx,yy], axis=2)
    rr2 = np.linalg.norm(rr, axis=-1)
    mask = ~(np.linalg.norm(rr, axis=-1) <  R)
    gq[mask] = 0
    gq[np.isnan(gq)] = 0 # NP freaks out if it's not defined...

    # Chirality delayed-evaluation
    ddens = d_vec.shape[0]     # Evaluate the cross correalation at these angle range
    chiral_vec = np.zeros(d_vec.shape) # Define vector to store 1-cross correltion
    for i, th_delay in enumerate(d_vec):

        # Rotated inverted positions
        irinv = rotate(posxinv, th_delay)
        # Interpolate
        gqinv = griddata((irinv[:, 0], irinv[:, 1]), pattern, (xx, yy), method=grid_method)
        gqinv[mask] = 0 # Ignore outside the circle
        gqinv[np.isnan(gqinv)] = 0 # NP freaking out

        o1m, o2m = np.average(gq), np.average(gqinv)
        o1norm, o2norm = np.sqrt(np.sum((gq-o1m)**2)), np.sqrt(np.sum((gqinv-o2m)**2))
        xcorr_local = ((gq-o1m)*(gqinv-o2m))/(o1norm*o2norm)
        chiral_vec[i] = 1-cross_correlation2D(gq, gqinv)

        # Print progress
        if i%int(ddens/3+1) == 0: print("\t Delay %10.2f deg" % th_delay, "(%4i of %5i, %5.0f%%)"%(i, ddens, 100*i/ddens), file=sys.stderr)
        # Plot current compare configuration
        if plot:
            # Plot original pattern, inverted-rotated one and local cross correlation
            fig, (ax, axmir, diff) = plt.subplots(1,3, dpi=300, figsize=(6,2))

            #clside = 0.02 # Good for strain in G/hBN
            clside = np.max(np.abs(pattern)) # Frame-wise
            #--------- Original image
            # Plot pattern
            ax.set_title('Original')
            ax.imshow(gq, aspect='equal', origin='lower', norm=Normalize(-clside, clside), cmap='RdBu')

            xside = gq.shape[0]/2
            # Mark moire position (from center of imshow (half the length of the grid) to moire orientation at grid_size/4
            ax.plot([xside, np.cos(tth*pi/180)*xside/2+xside], [xside, np.sin(tth*pi/180)*xside/2+xside],
                    'x:', lw=0.5, color='tab:green', zorder=11)
            # Mark mirror plane
            ax.vlines(xside, *ax.get_ylim(), ls='--', color='tab:gray', lw=0.5, zorder=11)
            ax.set_xticks([]) # Positions are note particularly informative here, I find
            ax.set_yticks([])

            #--------- Mirror image
            axmir.set_title('x-inv\nroto %.3gdeg' % th_delay)
            im = axmir.imshow(gqinv, aspect='equal', origin='lower', norm=Normalize(-clside, clside), cmap='RdBu')
            #plt.colorbar(im, ax=axmir, format='%8.2g') # Colorbar with pattern values, if you want it
            # Mark original moire
            axmir.plot([xside, np.cos(tth*pi/180)*xside/2+xside], [xside, np.sin(tth*pi/180)*xside/2+xside], 'x:', lw=0.5, color='tab:green', zorder=11)
            # Mirrored moire
            axmir.plot([xside, -np.cos((tth-th_delay)*pi/180)*xside/2+xside], [xside, np.sin((tth-th_delay)*pi/180)*xside/2+xside],
                       '+:', lw=0.5, color='tab:orange',  zorder=11)
            # Mirror plane
            axmir.vlines(xside, *axmir.get_ylim(), ls='--', color='tab:gray', lw=0.5, zorder=11)
            axmir.set_xticks([])
            axmir.set_yticks([])

            #--------- Cross-correlation image
            diff.set_title('Local \ncross corrlation')
            #clside = 2e-5 # Constant color range
            clside = np.max(np.abs(xcorr_local)) # Config-specific color range
            clside = 0.5*np.max(np.abs(xcorr_local)) # Saturated onfig-specific color range
            im = diff.imshow(xcorr_local, cmap='Spectral', norm=Normalize(-clside, clside), aspect='equal', origin='lower')
            plt.colorbar(im, ax=diff, format='%8.2g')
            diff.set_xticks([])
            diff.set_yticks([])
            plt.tight_layout()
            plt.show()

    # Plot 1-cross_corr as a function of angles d_vec
    if plot:
        chimin, thchimin, chimax, thchimax = min(chiral_vec), d_vec[np.argmin(chiral_vec)], max(chiral_vec), d_vec[np.argmax(chiral_vec)]
        print("chiral: min=%.3g (at %.3f deg) max=%.3g (at %.3g def)" % (chimin, thchimin, chimax, thchimax), file=sys.stderr)
        plt.plot(d_vec, chiral_vec, '.-')
        plt.plot(thchimin, chimin, 's', color='tab:red', label='min')
        plt.plot(thchimax, chimax, 'H', color='tab:green', label='max')
        plt.xlim([d_vec[0],d_vec[-1]])
        plt.ylim([-0.1, 1.5])
        plt.axhline(ls=':', lw=0.5, color='gray')
        plt.xlabel('angle')
        plt.ylabel('chirality = 1-XCorrelation')
        plt.show()

    return chiral_vec

if __name__ == '__main__':
    # Moire things
    acc, abn = 1.42039*np.sqrt(3), 1.44595702*np.sqrt(3) # Graphene REBO lattice constant, hBN Tersofshift lattice constant
    rho = abn/acc # Mismatch
    # Cutoff radius (in Agnstrom), ignore the outermost atoms (by an arbitrary margin)
    R = 133
    # Xyz filename
    index = sys.argv[2]
    # Printour header
    print('#angle_[deg] moire_angle_[deg] Chirality')
    # Chirality of each frame, do not plot steps
    if index == ':':
        print('Loading whole trajectory from %s' % sys.argv[1], file=sys.stderr)
        traj = ase_read(sys.argv[1], index=index)
        print('Loaded', file=sys.stderr)
        for index, geomASE in enumerate(traj):
            if index%int(len(traj)/10+1) == 0: print("Frame %4i of %5i (%5.0f%%)"%(index, len(traj), 100*index/len(traj)), file=sys.stderr)
            # Angle of flake, to tranform into moire angle. Only used for visualisation purposes
            th = pi/180*index*0.02 # FORWARD we can actually knwo from index
            #th = pi/180*(2-index*0.02) # BACKWARD
            tth = np.arctan((rho*np.sin(th))/(rho*np.cos(th)-1)) # Moire angle
            # Define delay angle array
            th0, th1, ddens = -0.5, 60.5, 150
            d_vec = np.linspace(th0, th1, ddens)
            print("Eval cross corr from th0=%.4f to th1=%.4f (%i steps)" % (d_vec[0], d_vec[-1], len(d_vec)), file=sys.stderr)
            # Get pos and pattern
            pos = geomASE.positions[:,:2] # Convert ase to 2D position
            pos -= geomASE.get_center_of_mass()[:2] # Mirror inversion are a bit screwd up if you are not in the center...
            pattern = geomASE.arrays['momenta'][:,0] # Strain should be the first component of velocity
            chiral_quant = np.min(chirality_xcorellation(pos, pattern, d_vec, R=R, plot=False, tth=tth*180/pi))
            print("%20.15f %20.15f %20.15f" % (th*180/pi, tth*180/pi, chiral_quant))
    # Chirality of single frame, plot steps
    else:
        index = int(sys.argv[2])
        print('Loading frame %i from trajectory %s' % (index, sys.argv[1]), file=sys.stderr)
        geomASE = ase_read(sys.argv[1], index=index)
        # Angle of the moire patterns. IN DEGREE. Only used for visualisation purposes
        th = pi/180*(index*0.02) # we can actually knwo from index (assuming is our AQS
        tth = np.arctan((rho*np.sin(th))/(rho*np.cos(th)-1)) # Moire
        print('Input angle %.2f moire angle %.2f' % (th*180/pi, tth*180/pi), file=sys.stderr)
        # Define delay angle array
        #d_vec = np.array([0, 60, 90, 180, 270])
        d_vec = np.array([0, -2*(90-tth*180/pi)])
        #d_vec = np.linspace(0,-90, 30)
        print("Eval cross corr from th0=%.4f to th1=%.4f (%i steps)" % (d_vec[0], d_vec[-1], len(d_vec)), file=sys.stderr)
        pos = geomASE.positions[:,:2] # 2D pos from ase geom
        pos -= geomASE.get_center_of_mass()[:2] # Mirror inversion are a bit screwd up if you are not in the center...
        pattern = geomASE.arrays['momenta'][:,0] # Strain should be the first component of velocity
        chiral_quant = np.min(chirality_xcorellation(pos, pattern, d_vec, R=R, plot=True, tth=tth*180/pi))
        print("%20.15f %20.15f %20.15f" % (th*180/pi, tth*180/pi, chiral_quant))
