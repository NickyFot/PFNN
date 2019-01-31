import os
import sys
import glob
import numpy as np

from scipy import ndimage
from scipy import misc

np.random.seed(2)

""" Options """

size = 128
sparsity = 15

""" Data """

heightmaps = 'data/heightmaps/hmap_*_smooth.txt'

""" Patches / Coordinates """

X, C = [], []

for fi, filename in enumerate(glob.glob(heightmaps)):
    
    H = np.loadtxt(filename)
    nsamples = (H.shape[0]*H.shape[1])//(sparsity*sparsity)
    
    print('Processing %s (%i x %i) %i [%i]' % (filename, H.shape[0], H.shape[1], len(X), nsamples))

    for si in range(nsamples):
        
        """ Random Location / Rotation"""
        
        xi, yi = np.random.randint(-size, H.shape[0]-size), np.random.randint(-size, H.shape[1]-size)
        r = np.degrees(np.random.uniform(-np.pi/2, np.pi/2))
        
        """ Find Patch """
        
        S = ndimage.interpolation.shift(H, (xi, yi), mode='reflect')[:size*2,:size*2]
        S = S[::-1, :] if np.random.uniform() > 0.5 else S
        S = S[:, ::-1] if np.random.uniform() > 0.5 else S
        S = ndimage.interpolation.rotate(S, r, reshape=False, mode='reflect')
        
        """ Extract Patch Area """
        
        P = S[size//2:size+size//2,size//2:size+size//2]
        
        """ Subtract Mean Value """
        
        P -= P.mean()
        
        """ Discard if height difference is too high """
        
        if np.any(abs(P) > 50): continue
        
        """ Add to List """
        
        X.append(P)
        C.append(np.array([fi, xi, yi, r]))

""" Save """
        
X = np.array(X).astype(np.float32)
C = np.array(C).astype(np.float32)

print(X.shape, C.shape)

np.savez_compressed('patches.npz', X=X, C=C)
