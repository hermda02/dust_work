import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
mpl.rcParams['text.usetex'] = True

CIB  = hp.read_map('CIB/CIB_857_n0064_60arcmin_temp.fits')
Td   = hp.read_map('v29/dust_hot_Td_c0001_k00100.fits')
mask = hp.read_map('CIB/CIB_mask_n0064.fits')

CIB  = CIB*mask
Td   = Td*mask

plt.scatter(CIB,Td,color='orange')
plt.xlim(-0.001,0.001)
plt.xlabel('CIB (MJy/sr)',size=15)
plt.ylabel('T_d (K)',size=15)
plt.ylim(5,25)
plt.show()


