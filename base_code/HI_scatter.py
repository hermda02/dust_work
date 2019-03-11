import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from scipy import stats
mpl.rcParams['text.usetex'] = True

#---------------------------------------------
#---------------------------------------------
h     = 6.626e-34
c     = 3.0e+8
k     = 1.38e-23

beta  = 1.70
Td    = 19.65

def planck(fre,T):
	return ((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))-1))

def MBB(fre,bet,T):
	return ((fre/(545*1e9))**bet)*((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))-1))	

#---------------------------------------------
#---------------------------------------------

HI     = hp.read_map('mom0_-90_90_1024.hpx.fits')
HI_vel = hp.read_map('ebv_lhd.hpx.fits')
I353   = hp.read_map('353-5_dust_2dust_v24.fits')
I545   = hp.read_map('545-1_dust_2dust_v24.fits')
I857   = hp.read_map('857-1_dust_2dust_v24.fits')
dust_T = hp.read_map('dust_hot_Td_c0001_k00029.fits')
beta_d = hp.read_map('dust_hot_beta_c0001_k00029.fits')
npix   = hp.nside2npix(64)
HI     = hp.ud_grade(HI,64)
mask   = np.empty(npix)

# Masking all points where column density is greater than 4x10^20 [cm^-2]
for i in range(npix):
	if (HI[i] < 4e20):
		mask[i] = 1.0
	else:
		mask[i] = 0.0

# Apply mask to all relevent maps to align pixels
HI     = HI*mask
I353   = I353*mask
I545   = I545*mask
I857   = I857*mask
dust_T = dust_T*mask
beta_d = beta_d*mask

# Remove all masked points from the arrays fo simplify binning/plotting
#-----------------------------------------------------------------
HI_new        = HI[np.where(HI != 0.0)[0]]
I353_new      = I353[np.where(I353 != 0.0)[0]]
I545_new      = I545[np.where(I545 != 0.0)[0]]
I857_new      = I857[np.where(I857 != 0.0)[0]]
Td_new        = dust_T[np.where(dust_T != 0.0)[0]]
beta_new      = beta_d[np.where(beta_d != 0.0)[0]]


norm_353      = planck(353*1e9,Td)
MBB_353       = MBB(353*1e9,beta,Td)
MBB_int_353   = I353_new/MBB_353
intensity_353 = I353_new/norm_353

norm_545      = planck(545*1e9,Td)
MBB_545       = MBB(545*1e9,beta,Td)
MBB_int_545   = I545_new/MBB_545
intensity_545 = I545_new/norm_545

norm_857      = planck(857*1e9,Td)
MBB_857       = MBB(857*1e9,beta,Td)
MBB_int_857   = I857_new/MBB_857
intensity_857 = I857_new/norm_857



dust_norm_353 = planck(353*1e9,Td_new)
dust_MBB_353  = MBB(353*1e9,beta_new,Td_new)
Td_MBB_353    = I353_new/dust_MBB_353
dust_int_353  = I353_new/dust_norm_353

I_NH_353      = I353_new/HI_new

dust_norm_545 = planck(545*1e9,Td_new)
dust_MBB_545  = MBB(545*1e9,beta_new,Td_new)
Td_MBB_545    = I545_new/dust_MBB_545
dust_int_545  = I545_new/dust_norm_545

I_NH_545      = I545_new/HI_new

dust_norm_857 = planck(857*1e9,Td_new)
dust_MBB_857  = MBB(857*1e9,beta_new,Td_new)
Td_MBB_857    = I857_new/dust_MBB_857
dust_int_857  = I857_new/dust_norm_857

I_NH_857      = I857_new/HI_new

#-----------------------------------------------------------------
#-----------------------------------------------------------------


xedges = np.arange(0,4,0.04)
xedges = xedges*1e20
yedges = np.arange(0,2,0.02)
yedges = yedges*1e15

aedges = np.arange(0,4,0.04)
aedges = aedges*1e-21
bedges = np.arange(10,25,0.15)

H,  xedges, yedges = np.histogram2d(HI_new, MBB_int_353, bins=(xedges,yedges),normed=True)
H2, xedges, yedges = np.histogram2d(HI_new, Td_MBB_353, bins=(xedges,yedges),normed=True)
H3, aedges, bedges = np.histogram2d(I_NH_545, Td_new, bins=(aedges,bedges),normed=True)

H  = H.T
H2 = H2.T
H3 = H3.T

X, Y  = np.meshgrid(xedges, yedges)
X2,Y2 = np.meshgrid(aedges, bedges)

plt.pcolormesh(X,Y,H,cmap='hot')
plt.xlabel(r'NH $\rm{[cm^{-2}]}$',size=15)
plt.ylabel(r'$I_{353}/B_{\nu}(19.65K)$',size=15)
plt.title(r'$H_I$ Column Density vs. Dust Emission',size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,3.75e20)
plt.ylim(0,0.6e15)
plt.show()

plt.pcolormesh(X,Y,H2,cmap='hot')
plt.xlabel(r'NH $\rm{[cm^{-2}]}$',size=15)
plt.ylabel(r'$I_{353}/B_{\nu}(T_{obs})$',size=15)
plt.title(r'$H_I$ Column Density vs. Dust Emission',size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,3.75e20)
plt.ylim(0,0.6e15)
plt.show()

plt.pcolormesh(X2,Y2,H3,cmap='hot')
plt.xlabel(r'$I_{545}/NH_I$', size=15)
plt.ylabel(r'$T_{obs}$', size=15)
plt.title('Intensity per Hydrogen vs. Dust Temperature',size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,3e-21)
plt.show()

# plt.scatter(I_NH,Td_new,color='orange',marker='+')
# plt.xlabel(r'$I_{545}/NH_I$', size=15)
# plt.ylabel(r'$T_{obs}$', size=15)
# plt.title('Intensity per Hydrogen vs. Dust Temperature',size=20)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlim(0,0.8e-20)
# plt.show()

# plt.scatter(HI,intensity)
# plt.xlim(-1e20,5e20)
# plt.ylim(-1e15,1e16)
# plt.show()


# plt.scatter(HI,dust_int)
# plt.xlim(-1e20,5e20)
# plt.ylim(-1e15,1e16)
# plt.show()