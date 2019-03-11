import numpy as np
import h5py
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as units
from scipy import stats
from astropy.coordinates import SkyCoord
mpl.rcParams['text.usetex'] = True

#------------------------------------------------------------------------------------------
# Planck function and MBB
#------------------------------------------------------------------------------------------
h     = 6.626e-34
c     = 3.0e+8
k     = 1.38e-23

beta  = 1.70
Td    = 19.65

def planck(fre,T):
	return ((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))-1))

def MBB(fre,bet,T):
	return ((fre/(545*1e9))**bet)*((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))-1))	

# Computing Planck 2015 MBB values
MBB_353            = MBB(353*1e9,beta,Td)
MBB_545            = MBB(545*1e9,beta,Td)
MBB_857            = MBB(857*1e9,beta,Td)

#------------------------------------------------------------------------------------------
# Reading maps
#------------------------------------------------------------------------------------------

I353_hot    = hp.read_map('../NHI/v36/sample_1000/dust_hot_band27_k00010.fits')
I545_hot    = hp.read_map('../NHI/v36/sample_1000/dust_hot_band31_k00010.fits')
I857_hot    = hp.read_map('../NHI/v36/sample_1000/dust_hot_band34_k00010.fits')

I353_cold   = hp.read_map('../NHI/v36/sample_1000/dust_cold_band27_k00010.fits')
I545_cold   = hp.read_map('../NHI/v36/sample_1000/dust_cold_band31_k00010.fits')
I857_cold   = hp.read_map('../NHI/v36/sample_1000/dust_cold_band34_k00010.fits')

dust_T_hot  = hp.read_map('../NHI/v36/sample_1000/dust_hot_Td_c0001_k01000.fits')
dust_b_hot  = hp.read_map('../NHI/v36/sample_1000/dust_hot_beta_c0001_k01000.fits')

dust_T_cold = hp.read_map('../NHI/v36/sample_1000/dust_cold_Td_c0001_k01000.fits')
dust_b_cold = hp.read_map('../NHI/v36/sample_1000/dust_cold_beta_c0001_k01000.fits')

CIB_mask    = hp.read_map('../CIB/CIB_mask_inverted_n0064.fits')

reddening   = hp.read_map('bayestar2017_n0064.fits')


#------------------------------------------------------------------------------------------
# Load the dust reddening map
#------------------------------------------------------------------------------------------

# f          = h5py.File('/media/daniel/DCHDrive/Data/dust_maps/bayestar/bayestar2017.h5','r')
# pix_info   = f['/pixel_info'][:]
# EBV_total  = np.median(f['/samples'][:,:,-1],axis=1)
# f.close()

# nside_max  = np.max(pix_info['nside'])

# for nside in np.unique(pix_info['nside']):
# 	# Get indices of all pixels at current nside level
#     idx = pix_info['nside'] == nside

#     # Extract E(B-V) of each selected pixel
#     pix_val_n = EBV_total[idx]

#     # Determine nested index of each selected pixel in upsampled map
#     mult_factor = (nside_max/nside)**2
#     pix_idx_n = pix_info['healpix_index'][idx] * mult_factor

#     # Write the selected pixels into the upsampled map
#     for offset in range(mult_factor):
#         pix_val[pix_idx_n+offset] = pix_val_n[:]

# hp.write_map('/media/daniel/DCHDrive/Data/dust_maps/bayestar/bayestar2017_n1024.fits',pix_val,nest=True)
# exit()

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

npix       = hp.nside2npix(64)
pix_val    = np.empty(npix, dtype='f8')
pix_val[:] = np.nan

A_v_bayestar    = np.empty(npix,dtype='f8')
A_v_bayestar[:] = np.nan


for i in range(64*64*12):
	if (CIB_mask[i] == 0.0):
		CIB_mask[i] = np.nan

reddening = reddening * CIB_mask

for i in range(64*64*12):
	if (reddening[i] == -1.6375e+30):
		reddening[i] = np.nan

	# if (reddening[i] < 0.05):
	# 	reddening[i] = np.nan

	if (np.isnan(reddening[i])):
		I353_hot[i]    = np.nan
		I353_cold[i]   = np.nan
		I545_hot[i]    = np.nan
		I545_cold[i]   = np.nan
		I857_hot[i]    = np.nan
		I857_cold[i]   = np.nan

		dust_T_hot[i]  = np.nan
		dust_T_cold[i] = np.nan
		dust_b_hot[i]  = np.nan
		dust_b_cold[i] = np.nan

	else:
		# print(pix_val[i])
		A_v_bayestar[i] = 2.742*reddening[i]

#------------------------------------------------------------------------------------------

A_v_bayestar_new = A_v_bayestar[np.where(True != np.isnan(A_v_bayestar))]

I353_new_hot     = I353_hot[np.where(True != np.isnan(I353_hot))]
I545_new_hot     = I545_hot[np.where(True != np.isnan(I545_hot))]
I857_new_hot     = I857_hot[np.where(True != np.isnan(I857_hot))]
dust_T_new_hot   = dust_T_hot[np.where(True != np.isnan(dust_T_hot))]
dust_b_new_hot   = dust_b_hot[np.where(True != np.isnan(dust_b_hot))]
 
I353_new_cold    = I353_cold[np.where(True != np.isnan(I353_cold))]
I545_new_cold    = I545_cold[np.where(True != np.isnan(I545_cold))]
I857_new_cold    = I857_cold[np.where(True != np.isnan(I857_cold))]
dust_T_new_cold  = dust_T_cold[np.where(True != np.isnan(dust_T_cold))]
dust_b_new_cold  = dust_b_cold[np.where(True != np.isnan(dust_b_cold))]

I353_total       = I353_new_hot + I353_new_cold
I545_total       = I545_new_hot + I545_new_cold
I857_total       = I857_new_hot + I857_new_cold

# Computing I/MBB from Planck 2015
MBB_int_353_hot    = I353_new_hot/MBB_353
MBB_int_545_hot    = I545_new_hot/MBB_545
MBB_int_857_hot    = I857_new_hot/MBB_857

MBB_int_353_cold   = I353_new_cold/MBB_353
MBB_int_545_cold   = I545_new_cold/MBB_545
MBB_int_857_cold   = I857_new_cold/MBB_857

MBB_int_353_tot    = I353_total/MBB_353
MBB_int_545_tot    = I545_total/MBB_545
MBB_int_857_tot    = I857_total/MBB_857

# Computing I/B_v from my analysis
MBB_model_353_hot  = MBB(353*1e9,dust_b_new_hot,dust_T_new_hot)
MBB_model_545_hot  = MBB(545*1e9,dust_b_new_hot,dust_T_new_hot)
MBB_model_857_hot  = MBB(857*1e9,dust_b_new_hot,dust_T_new_hot)

MBB_model_353_cold = MBB(353*1e9,dust_b_new_cold,dust_T_new_cold)
MBB_model_545_cold = MBB(545*1e9,dust_b_new_cold,dust_T_new_cold)
MBB_model_857_cold = MBB(857*1e9,dust_b_new_cold,dust_T_new_cold)

# Computing I/MBB from analysis
I_MBB_353_hot      = I353_new_hot/MBB_model_353_hot
I_MBB_545_hot      = I545_new_hot/MBB_model_545_hot
I_MBB_857_hot      = I857_new_hot/MBB_model_857_hot

I_MBB_353_cold     = I353_new_cold/MBB_model_353_cold
I_MBB_545_cold     = I545_new_cold/MBB_model_545_cold
I_MBB_857_cold     = I857_new_cold/MBB_model_857_cold

MBB_353_total      = MBB_model_353_hot + MBB_model_353_cold
MBB_545_total      = MBB_model_545_hot + MBB_model_545_cold
MBB_857_total      = MBB_model_857_hot + MBB_model_857_cold

I_MBB_353          = I353_total/MBB_353_total
I_MBB_545          = I545_total/MBB_545_total
I_MBB_857          = I857_total/MBB_857_total

#----------------------------------------------------------------
#----------------------------------------------------------------
new_545 = []
new_A_v = []
for i in range(13938):
	if (MBB_int_545_tot[i] < 2.5e20):
		new_545.append(MBB_int_545_tot[i])
		new_A_v.append(A_v_bayestar_new[i])

new_545_2 = []
new_A_v_2 = []

for i in range(13938):
	if (I_MBB_545[i] < 2.5e20):
		new_545_2.append(I_MBB_545[i])
		new_A_v_2.append(A_v_bayestar_new[i])


slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(new_A_v,new_545)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(new_A_v_2,new_545_2)

xedges = np.logspace(-1,1,num=100)
yedges = np.logspace(-3,0,num=100)
yedges = yedges*3e20

H,  xedges, yedges  = np.histogram2d(A_v_bayestar_new, MBB_int_545_tot, bins=(xedges,yedges),normed=True)
H2, xedges, yedges  = np.histogram2d(A_v_bayestar_new, I_MBB_545, bins=(xedges,yedges),normed=True)

H       = H.T
H2      = H2.T

X, Y  = np.meshgrid(xedges, yedges)

print(std_err1)

plt.pcolormesh(X,Y,H,cmap='hot')
# plt.plot(xedges,slope1*xedges+intercept1,color='white')
plt.xlabel(r'A_V',size=15)
plt.ylabel(r'$I_{545}/B_{\nu}(19.65K)$',size=15)
plt.title(r'$A_V$ vs. Dust Emission',size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1,6.0)
plt.ylim(1e18,3e20)
plt.show()

print(std_err2)

plt.pcolormesh(X,Y,H2,cmap='hot')
# plt.plot(xedges,slope2*xedges+intercept2,color='white')
plt.xlabel(r'A_V',size=15)
plt.ylabel(r'$I_{545}/B_{\nu}(T_{obs})$',size=15)
plt.title(r'$A_V$ vs. Dust Emission',size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1,6.0)
plt.ylim(1e18,3e20)
plt.show()