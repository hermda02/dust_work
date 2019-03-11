import numpy as np
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
npix       = hp.nside2npix(64)
pix_val    = np.empty(npix, dtype='f8')
pix_val[:] = np.nan

A_v_bayestar    = np.empty(npix,dtype='f8')
A_v_bayestar[:] = np.nan

for i in range(64*64*12):
	if (CIB_mask[i] == 0.0):
		CIB_mask[i] = np.nan

# reddening = reddening * CIB_mask

for i in range(64*64*12):
	if (reddening[i] == -1.6375e+30):
		reddening[i] = np.nan

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

total_int_353 = I353_hot + I353_cold
total_int_545 = I545_hot + I545_cold
total_int_857 = I857_hot + I857_cold

model_353_hot  = MBB(353*1e9,dust_b_hot,dust_T_hot)
model_545_hot  = MBB(545*1e9,dust_b_hot,dust_T_hot)
model_857_hot  = MBB(857*1e9,dust_b_hot,dust_T_hot)

model_353_cold = MBB(353*1e9,dust_b_cold,dust_T_cold)
model_545_cold = MBB(545*1e9,dust_b_cold,dust_T_cold)
model_857_cold = MBB(857*1e9,dust_b_cold,dust_T_cold)

total_model_353 = model_353_cold + model_353_hot
total_model_545 = model_545_cold + model_545_hot
total_model_857 = model_857_cold + model_857_hot

new_map_353_1 = total_int_353/A_v_bayestar*MBB_353
new_map_545_1 = total_int_545/A_v_bayestar*MBB_545
new_map_857_1 = total_int_857/A_v_bayestar*MBB_857

new_map_353_2 = total_int_353/A_v_bayestar*total_model_353
new_map_545_2 = total_int_545/A_v_bayestar*total_model_545
new_map_857_2 = total_int_857/A_v_bayestar*total_model_857

#------------------------------------------------------------------------------------------

hp.mollview(new_map_857_1,min=5e-16,max=1e-13,fig=1)
plt.title(r'857GHz $I_{\nu}/E(B-V)*B_{\nu}(19.6K)$')

hp.mollview(new_map_857_2,min=5e-16,max=1e-13,fig=2)
plt.title(r'857GHz $I_{\nu}/E(B-V)*B_{\nu}(19.6K)$')
plt.show()

# hp.mollview(new_map_545_2-new_map_545_1,min=-1e20,max=1e20,fig=3)
# plt.title(r'545GHz $I_{\nu}/B_{\nu}(T_{obs})*E(B-V) - $I_{\nu}/B_{\nu}(19.6K)*E(B-V)$')
# plt.show()