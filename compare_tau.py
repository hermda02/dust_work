import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import stats

mpl.rcParams['text.usetex'] = True

# For beautiful maps
planck  = np.loadtxt('/home/daniel/GraduateSchool/MastersThesis/tools/Planck_color.txt')/255.
pl_cmap = colors.LinearSegmentedColormap.from_list('planck',planck)

#------------------------------------------------------------------------------------------
# Initialize maps
#------------------------------------------------------------------------------------------

print('==================================')
version = raw_input("Which version (v45)? ")
version = str(version)


# Commander Two-component dust model 
I353_hot     = hp.read_map('dust_data/2dust/'+version+'/dust_hot_band27_k00010.fits',verbose=False)
I545_hot     = hp.read_map('dust_data/2dust/'+version+'/dust_hot_band31_k00010.fits',verbose=False)
I857_hot     = hp.read_map('dust_data/2dust/'+version+'/dust_hot_band34_k00010.fits',verbose=False)
I3000_hot    = hp.read_map('dust_data/2dust/'+version+'/dust_hot_band49_k00010.fits',verbose=False)

I353_cold    = hp.read_map('dust_data/2dust/'+version+'/dust_cold_band27_k00010.fits',verbose=False)
I545_cold    = hp.read_map('dust_data/2dust/'+version+'/dust_cold_band31_k00010.fits',verbose=False)
I857_cold    = hp.read_map('dust_data/2dust/'+version+'/dust_cold_band34_k00010.fits',verbose=False)
I3000_cold   = hp.read_map('dust_data/2dust/'+version+'/dust_cold_band49_k00010.fits',verbose=False)

dust_T_hot   = hp.read_map('dust_data/2dust/'+version+'/dust_hot_Td_c0002_k00010.fits',verbose=False)
dust_b_hot   = hp.read_map('dust_data/2dust/'+version+'/dust_hot_beta_c0002_k00010.fits',verbose=False)

dust_T_cold  = hp.read_map('dust_data/2dust/'+version+'/dust_cold_Td_c0002_k00010.fits',verbose=False)
dust_b_cold  = hp.read_map('dust_data/2dust/'+version+'/dust_cold_beta_c0002_k00010.fits',verbose=False)


# Planck 2015 Dust Data
I_545_planck = hp.read_map('/home/daniel/data/planck_dust/I_545_MJy_dust-commander_0064_R2.fits',verbose=False)

# tau_planck   = hp.read_map('dust_data/CompMap_tau_60arcmin_0064.fits',verbose=False)
Td_planck    = hp.read_map('/home/daniel/data/planck_dust/Td_mean_dust-commander_0064_R2.fits',verbose=False)
beta_planck  = hp.read_map('/home/daniel/data/planck_dust/beta_mean_dust-commander_0064_R2.fits',verbose=False)


# Commander 2018 T_d map
commander_T  = hp.read_map('/home/daniel/data/planck_dust/commander_dx12_dust_T_n0064_60arc.fits',verbose=False)

# GNILC -- intensity maps are in uK_cmb
I_353_gnilc  = hp.read_map('/home/daniel/data/gnilc/COM_CompMap_Dust-GNILC-F353_60arcmin_0064_R2.00.fits',verbose=False)
I_545_gnilc  = hp.read_map('/home/daniel/data/gnilc/COM_CompMap_Dust-GNILC-F545_60arcmin_0064_R2.00.fits',verbose=False)
I_857_gnilc  = hp.read_map('/home/daniel/data/gnilc/COM_CompMap_Dust-GNILC-F857_60arcmin_0064_R2.00.fits',verbose=False)

tau_gnilc    = hp.read_map('/home/daniel/data/gnilc/COM_CompMap_Dust-GNILC-Model-Opacity_60arcmin_0064_R2.00.fits',verbose=False)
beta_gnilc   = hp.read_map('/home/daniel/data/gnilc/COM_CompMap_Dust-GNILC-Model-Spectral-Index_60arcmin_0064_R2.00.fits',verbose=False)
Td_gnilc     = hp.read_map('/home/daniel/data/gnilc/COM_CompMap_Dust-GNILC-Model-Temperature_60arcmin_0064_R2.00.fits',verbose=False)


# premise 2019
tau_prem_353 = hp.read_map('/home/daniel/data/premise/tauNoMaskNoThresh_n0064_60arcmin.fits',verbose=False)
beta_prem    = hp.read_map('/home/daniel/data/premise/betaNoMaskNoThresh_n0064_60arcmin.fits',verbose=False)
Td_prem      = hp.read_map('/home/daniel/data/premise/tempNoMaskNoThresh_n0064_60arcmin.fits',verbose=False)


# Meisner/Finkbeiner - 2014
tau_mf_545   = hp.read_map('/home/daniel/data/meisner-finkbeiner/meisner_finkbeiner_tau_545_0064_masked.fits',verbose=False)
t2_mf        = hp.read_map('/home/daniel/data/meisner-finkbeiner/meisner_finkbeiner_T2_0064.fits',verbose=False)


# HI/reddening surveys (Hensley velocity filtered,HI4PI,Bayestar2017)
HI           = hp.read_map('dust_data/HI_vel_filter_60arcmin_0064.fits',verbose=False)
HI4PI        = hp.read_map('dust_data/NHI4PI_60arc_0064.fits',verbose=False)

reddening    = hp.read_map('dust_data/bayestar2017_60arcmin_n0064.fits',verbose=False)
redz         = hp.read_map('dust_data/bayestar2017_n0064.fits',verbose=False)

#------------------------------------------------------------------------------------------
# Initialize values/functions
#------------------------------------------------------------------------------------------
npix            = hp.nside2npix(64)

h     = 6.626e-34     # Planck constant, J s
c     = 3.0e+8        # Speed of light, m/s
k     = 1.38e-23      # Boltzmann constant, J/K

beta_mf = 2.82
beta  = 1.70
Td    = 19.65

def planck(fre,T):
	return ((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))-1))


# Computing Planck 2015 MBB values
I_545_planck    = I_545_planck*1e-20
tau_planck      = I_545_planck/planck(545*1e9,Td_planck)
I_353_planck    = tau_planck*(353./545.)**beta_planck*planck(353*1e9,Td_planck)
I_857_planck    = tau_planck*(857./545.)**beta_planck*planck(857*1e9,Td_planck)
I_3000_planck   = tau_planck*(2998./545.)**beta_planck*planck(2998*1e9,Td_planck)


# Computing GNILC extrapolations
I_353_gnilc     = I_353_gnilc*1e-20
I_545_gnilc     = tau_gnilc*(545./353.)**beta_gnilc*planck(545*1e9,Td_gnilc)
I_857_gnilc     = tau_gnilc*(857./353.)**beta_gnilc*planck(857*1e9,Td_gnilc)
I_3000_gnilc    = tau_gnilc*(2998./353.)**beta_gnilc*planck(2998*1e9,Td_gnilc)


# Computing permise 2019 MBB extrapolations
tau_prem_545    = tau_prem_353*(545./353.)**beta_prem
tau_prem_857    = tau_prem_353*(857./353.)**beta_prem
tau_prem_3000   = tau_prem_353*(2998./353.)**beta_prem


# Extrapolating tau_mf to other frequencies

tau_mf_353      = tau_mf_545*(353./545.)**beta_mf
tau_mf_857      = tau_mf_545*(857./545.)**beta_mf
tau_mf_3000     = tau_mf_545*(2998./545.)**beta_mf


# Compensating for smoothing issues around the mask

for i in range(npix):
	if ((redz[i]-reddening[i]) > 1.0):
		reddening[i] = np.nan

A_v_bayestar    = np.empty(npix,dtype='f8')
HI_masked       = np.empty(npix,dtype='f8')
HI_newz         = np.empty(npix,dtype='f8')
A_v_bayestar[:] = np.nan

HI              = HI/1e20

tau_planck_545  = (I_545_planck)/planck(545*1e9,19.6)
tau_gnilc_545   = tau_gnilc*(545./353.)**beta_gnilc

#------------------------------------------------------------------------------------------
# Converting intensities (uK_cmb/MJy sr^-1) to W sr^-1 m^-2 Hz^-1
#------------------------------------------------------------------------------------------

I_nu_hot        = np.empty((4,npix))
I_nu_cold       = np.empty((4,npix))

I_nu_hot[0][:]  = I353_hot*0.29008843E-23
I_nu_cold[0][:] = I353_cold*0.29008843E-23

I353_hot        = I353_hot*0.29008843E-23
I353_cold       = I353_cold*0.29008843E-23

I_nu_hot[1][:]  = I545_hot*0.57287522E-24
I_nu_cold[1][:] = I545_cold*0.57287522E-24

I545_hot        = I545_hot*0.57287522E-24
I545_cold       = I545_cold*0.57287522E-24

I_nu_hot[2][:]  = I857_hot*1e-20
I_nu_cold[2][:] = I857_cold*1e-20

I857_hot        = I857_hot*1e-20
I857_cold       = I857_cold*1e-20

I_nu_hot[3][:]  = I3000_hot*1e-20
I_nu_cold[3][:] = I3000_cold*1e-20

I3000_hot       = I3000_hot*1e-20
I3000_cold      = I3000_cold*1e-20


#------------------------------------------------------------------------------------------
# Calculate \tau_\nu
#------------------------------------------------------------------------------------------

# Commander Model \tau_\nu
# ------------------------------------------------------------
	# [0][:] is the two-component Commander output           |
	# [1][:] is the Planck 2015 paper \tau results           |
	# [2][:] is the Planck 2015 intensity divided by Bnu(19K)|
	# [3][:] is the 2015 GNILC \tau results                  |
	# [4][:] is the Meisner-Finkbeiner results               |
	# [5][:] is the premise results (2019)                   |
# ------------------------------------------------------------

tau_HI_nu  = np.empty((6,npix))
tau_ebv_nu = np.empty((6,npix))
model      = ''
again      = 'y'

def initialize():

	global model
	global tau_HI_nu
	global tau_ebv_nu

	tau_HI_nu  = np.empty((6,npix))
	tau_ebv_nu = np.empty((6,npix))

	model = raw_input('353, 545, 857, or 3000GHz? ')

	if (model == '353'):
		tau_HI_nu[0][:]  = I_nu_hot[0][:]/planck(353*1e9,dust_T_hot) + I_nu_cold[0][:]/planck(353*1e9,dust_T_cold) 
		tau_ebv_nu[0][:] = I_nu_hot[0][:]/planck(353*1e9,dust_T_hot) + I_nu_cold[0][:]/planck(353*1e9,dust_T_cold) 
		tau_HI_nu[1][:]  = tau_planck*(353./545.)**beta_planck
		tau_ebv_nu[1][:] = tau_HI_nu[1][:]
		tau_HI_nu[2][:]  = (I_353_planck)/planck(353*1e9,19.6)
		tau_ebv_nu[2][:] = tau_HI_nu[2][:]
		tau_HI_nu[3][:]  = tau_gnilc
		tau_ebv_nu[3][:] = tau_gnilc
		tau_HI_nu[4][:]  = tau_mf_353
		tau_ebv_nu[4][:] = tau_mf_353
		tau_HI_nu[5][:]  = tau_prem_353
		tau_ebv_nu[5][:] = tau_prem_353


	if (model == '545'):
		tau_HI_nu[0][:]  = I_nu_hot[1][:]/planck(545*1e9,dust_T_hot) + I_nu_cold[1][:]/planck(545*1e9,dust_T_cold)
		tau_ebv_nu[0][:] = tau_HI_nu[0][:]
		tau_HI_nu[1][:]  = tau_planck
		tau_ebv_nu[1][:] = tau_planck
		tau_HI_nu[2][:]  = (I_545_planck)/planck(545*1e9,19.6)
		tau_ebv_nu[2][:] = tau_HI_nu[2][:]
		tau_HI_nu[3][:]  = tau_gnilc*(545./353.)**beta_gnilc
		tau_ebv_nu[3][:] = tau_HI_nu[3][:]
		tau_HI_nu[4][:]  = tau_mf_545
		tau_ebv_nu[4][:] = tau_mf_545
		tau_HI_nu[5][:]  = tau_prem_545
		tau_ebv_nu[5][:] = tau_prem_545

	if (model == '857'):
		tau_HI_nu[0][:]  = I_nu_hot[2][:]/planck(857*1e9,dust_T_hot) + I_nu_cold[2][:]/planck(857*1e9,dust_T_cold)
		tau_ebv_nu[0][:] = tau_HI_nu[0][:]
		tau_HI_nu[1][:]  = tau_planck*(857./545.)**beta_planck
		tau_ebv_nu[1][:] = tau_HI_nu[1][:]
		tau_HI_nu[2][:]  = (I_857_planck)/planck(857*1e9,19.6)
		tau_ebv_nu[2][:] = tau_HI_nu[2][:]
		tau_HI_nu[3][:]  = tau_gnilc*(857./353.)**beta_gnilc
		tau_ebv_nu[3][:] = tau_HI_nu[3][:]
		tau_HI_nu[4][:]  = tau_mf_857
		tau_ebv_nu[4][:] = tau_mf_857
		tau_HI_nu[5][:]  = tau_prem_857
		tau_ebv_nu[5][:] = tau_prem_857

	if (model == '3000'):
		tau_HI_nu[0][:]  = I_nu_hot[3][:]/planck(2998*1e9,dust_T_hot) + I_nu_cold[3][:]/planck(2998*1e9,dust_T_cold)
		tau_ebv_nu[0][:] = tau_HI_nu[0][:]
		tau_HI_nu[1][:]  = tau_planck*(2998./545.)**beta_planck
		tau_ebv_nu[1][:] = tau_HI_nu[1][:]
		tau_HI_nu[2][:]  = (I_3000_planck)/planck(2998*1e9,19.6)
		tau_ebv_nu[2][:] = tau_HI_nu[2][:]
		tau_HI_nu[3][:]  = tau_gnilc*(3000./353.)**beta_gnilc
		tau_ebv_nu[3][:] = tau_HI_nu[3][:]
		tau_HI_nu[4][:]  = tau_mf_3000
		tau_ebv_nu[4][:] = tau_mf_3000
		tau_HI_nu[5][:]  = tau_prem_3000
		tau_ebv_nu[5][:] = tau_prem_3000

#==========================================================================================
# HI and E(B-V) plotting
#==========================================================================================

# HI functions
# -----------------------------
def HI_maps():

	# newmap = np.empty(npix)

	# for i in range(npix):
	# 	if (np.log10(tau_HI_nu[4][i]/HI[i]) > -23.8):
	# 		newmap[i] = np.nan
	# 	else:
	# 		newmap[i] = tau_HI_nu[4][i]

	# hp.write_map('/home/daniel/Data/meisner-finkbeiner/meisner_finkbeiner_tau_545_0064_masked.fits',newmap)

	# hp.mollview(np.log10(tau_HI_nu[4][:]/HI),max=-23.8,cmap=pl_cmap)
	# plt.show()
	# exit()

	fig, (((ax1), (ax2),(ax3)), ((ax4), (ax5), (ax6))) = plt.subplots(2,3)

	if (model == '353'):
		fig.suptitle(r'$\tau_{353}$/N ${\rm H_I}$', size=20)
		file = 'tau_353_HI'
		mini = -27
		maxi = -24

	if (model == '545'):
		fig.suptitle(r'$\tau_{545}$/N ${\rm H_I}$', size=20)
		file = 'tau_545_HI'
		mini = -27
		maxi = -24

	if (model == '857'):
		fig.suptitle(r'$\tau_{857}$/N ${\rm H_I}$', size=20)
		file = 'tau_857_HI'
		mini = -26
		maxi = -24

	if (model == '3000'):
		fig.suptitle(r'$\tau_{3000}$/N ${\rm H_I}$', size=20)
		file = 'tau_3000_HI'
		mini = -25
		maxi = -23

	for i in range(npix):
		if (np.isnan(HI[i]) == True):
			print('AHH')
			print(i)
			exit()

	map1 = tau_HI_nu[0][:]/(HI*1e20)
	map2 = tau_HI_nu[1][:]/(HI*1e20)
	map3 = tau_HI_nu[2][:]/(HI*1e20)
	map4 = tau_HI_nu[3][:]/(HI*1e20)
	map5 = tau_HI_nu[4][:]/(HI*1e20)
	map6 = tau_HI_nu[5][:]/(HI*1e20)

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')
	ax5.axis('off')
	ax6.axis('off')

	plt.sca(ax1)
	hp.mollview(np.log10(map1),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'Two-Component Dust MBB')
	plt.sca(ax2)
	hp.mollview(np.log10(map2),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'Planck 2015 MBB')
	plt.sca(ax3)
	hp.mollview(np.log10(map3),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'$B_{\nu}(19.6K)$')
	plt.sca(ax4)
	hp.mollview(np.log10(map4),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'GNILC MBB')
	plt.sca(ax5)
	hp.mollview(np.log10(map5),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'M\&F 2014')
	plt.sca(ax6)
	hp.mollview(np.log10(map6),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'\texttt{premise]')
	# plt.savefig('tau/'+version+'/nhi/'+file, dpi=1000,bbox_inches='tight')
	plt.show()
	plt.close()

def HI_scatter():

	dummy1 = tau_HI_nu[0][:]
	dummy2 = tau_HI_nu[1][:]
	dummy3 = tau_HI_nu[2][:]
	dummy4 = tau_HI_nu[3][:]
	dummy5 = tau_HI_nu[4][:]
	dummy6 = tau_HI_nu[5][:]

	for i in range(npix):
		if (HI[i] > 4.0):
			HI_masked[i]  = np.nan
			dummy1[i]     = np.nan
			dummy2[i]     = np.nan
			dummy3[i]     = np.nan
			dummy4[i]     = np.nan
			dummy5[i]     = np.nan
			dummy6[i]     = np.nan
		else:
			HI_masked[i]  = HI[i]

	tau_HI_2d  = dummy1[np.where(True != np.isnan(dummy1))]
	tau_HI_pl  = dummy2[np.where(True != np.isnan(HI_masked))]
	tau_HI_gn  = dummy4[np.where(True != np.isnan(HI_masked))]
	tau_HI_bnu = dummy3[np.where(True != np.isnan(HI_masked))]
	tau_HI_mf  = dummy5[np.where(True != np.isnan(dummy5))]
	HI_mf      = HI_masked[np.where(True != np.isnan(dummy5))]
	tau_HI_pre = dummy6[np.where(True != np.isnan(HI_masked))]
	HI_new     = HI_masked[np.where(True != np.isnan(dummy1))]

	fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2,6,figsize=(12,4),sharex=True,sharey='row')


	if (model == '353'):
		fig.suptitle(r'HI Column Density vs. $\tau_{353}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1
		yedges1 = np.arange(0,1.2,0.012)
		yedges1 = yedges1*1e-5
		ymin    = 0
		ylim    = 6e-6
		file    = 'HI_scatter_353' 

	if (model == '545'):
		fig.suptitle(r'HI Column Density vs. $\tau_{545}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1
		yedges1 = np.arange(0,1.5,0.015)
		yedges1 = yedges1*1e-5
		ymin    = 0
		ylim    = 1e-5
		file    = 'HI_scatter_545' 

	if (model == '857'):
		fig.suptitle(r'HI Column Density vs. $\tau_{857}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1
		yedges1 = np.arange(0,25,0.025)
		yedges1 = yedges1*1e-5
		ymin    = 0
		ylim    = 1.5e-5
		file    = 'HI_scatter_857' 

	if (model == '3000'):
		fig.suptitle(r'HI Column Density vs. $\tau_{3000}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1
		yedges1 = np.arange(0,25,0.25)
		yedges1 = yedges1*1e-5	
		ymin    = 1e-5
		ylim    = 2.5e-4
		file    = 'HI_scatter_3000' 

	# Current two-component
	# -----------------------------------------------------------------------------------------
	H1, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_2d, bins=(xedges1,yedges1),normed=True)
	H1     = H1.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(HI_new,tau_HI_2d)

	mean1,edges, nums = stats.binned_statistic(HI_new,tau_HI_2d,bins=xedges1,statistic='mean')
	std1, edges, nums = stats.binned_statistic(HI_new,tau_HI_2d,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# Planck 2015
	# -----------------------------------------------------------------------------------------
	H2, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_pl, bins=(xedges1,yedges1),normed=True)
	H2     = H2.T

	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(HI_new,tau_HI_pl)

	mean2,edges, nums = stats.binned_statistic(HI_new,tau_HI_pl,bins=xedges1,statistic='mean')
	std2, edges, nums = stats.binned_statistic(HI_new,tau_HI_pl,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# GNILC
	# -----------------------------------------------------------------------------------------
	H3, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_gn, bins=(xedges1,yedges1),normed=True)
	H3     = H3.T

	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(HI_new,tau_HI_gn)

	mean3,edges, nums = stats.binned_statistic(HI_new,tau_HI_gn,bins=xedges1,statistic='mean')
	std3, edges, nums = stats.binned_statistic(HI_new,tau_HI_gn,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# I_Planck/B_nu(19.6K)
	# -----------------------------------------------------------------------------------------
	H4, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_bnu, bins=(xedges1,yedges1),normed=True)
	H4     = H4.T

	slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(HI_new,tau_HI_bnu)

	mean4,edges, nums = stats.binned_statistic(HI_new,tau_HI_bnu,bins=xedges1,statistic='mean')
	std4, edges, nums = stats.binned_statistic(HI_new,tau_HI_bnu,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# Meisner/Finkbeiner 2014
	# -----------------------------------------------------------------------------------------
	H5, xedges1, yedges1 = np.histogram2d(HI_mf,tau_HI_mf, bins=(xedges1,yedges1),normed=True)
	H5     = H5.T

	slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(HI_mf,tau_HI_mf)

	mean5,edges, nums = stats.binned_statistic(HI_mf,tau_HI_mf,bins=xedges1,statistic='mean')
	std5, edges, nums = stats.binned_statistic(HI_mf,tau_HI_mf,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# premise 2019
	# -----------------------------------------------------------------------------------------
	H6, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_pre, bins=(xedges1,yedges1),normed=True)
	H6     = H6.T

	slope6, intercept6, r_value6, p_value6, std_err6 = stats.linregress(HI_new,tau_HI_pre)

	mean6,edges, nums = stats.binned_statistic(HI_new,tau_HI_pre,bins=xedges1,statistic='mean')
	std6, edges, nums = stats.binned_statistic(HI_new,tau_HI_pre,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------

	centers = (edges[:-1] + edges[1:])/2


	fig.text(0.5, 0.02, r'${\rm N_H 10^{20} [cm^{-2}]}$', ha='center', size=12)

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	# ax1.plot(xedges1,np.poly1d(np.polyfit(HI_new_new,tau_HI_2d,1))(xedges1))
	ax1.plot(xedges1,slope1*xedges1+intercept1, color='black', linestyle='--')
	ax1.set_title(r'Two MMBs',size=8)
	ax1.set_ylabel(r'$\tau$', size=25)
	ax1.set_ylim([ymin,ylim])
	ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	ax2.pcolormesh(X1,Y1,H2,cmap=plt.get_cmap('hot_r'))
	ax2.plot(xedges1,slope2*xedges1+intercept2, color='black', linestyle='--')
	ax2.set_title(r'Planck 2015',size=8)

	ax3.pcolormesh(X1,Y1,H3,cmap=plt.get_cmap('hot_r'))
	ax3.plot(xedges1,slope3*xedges1+intercept3, color='black', linestyle='--')
	ax3.set_title(r'GNILC',size=8)

	ax4.pcolormesh(X1,Y1,H4,cmap=plt.get_cmap('hot_r'))
	ax4.plot(xedges1,slope4*xedges1+intercept4, color='black', linestyle='--')
	ax4.set_title(r'$B_{\nu}(19.6K)$',size=8)

	ax5.pcolormesh(X1,Y1,H5,cmap=plt.get_cmap('hot_r'))
	ax5.plot(xedges1,slope5*xedges1+intercept5, color='black', linestyle='--')
	ax5.set_title(r'Meisner/Finkbeiner',size=8)

	ax6.pcolormesh(X1,Y1,H6,cmap=plt.get_cmap('hot_r'))
	ax6.plot(xedges1,slope6*xedges1+intercept6, color='black', linestyle='--')
	ax6.set_title(r'\texttt{premise}',size=8)

	ax7.plot(centers,std1/mean1,'k-')
	ax7.set_ylabel(r'$\sigma/\mu$', size=15)
	ax7.tick_params(axis='both',labelsize=12)
	ax7.ticklabel_format(axis='x',style='sci')
	ax7.set_xlim([0.5,3.75])
	ax7.set_ylim([0,1])

	ax8.plot(centers,std2/mean2,'k-')
	ax9.plot(centers,std3/mean3,'k-')
	ax10.plot(centers,std4/mean4,'k-')
	ax11.plot(centers,std5/mean5,'k-')
	ax12.plot(centers,std6/mean6,'k-')
	
	plt.subplots_adjust(wspace=0.0)
	plt.subplots_adjust(hspace=0.1)

	plt.savefig('tau/'+version+'/nhi/'+file, dpi=300,bbox_inches='tight')
	# plt.show()
	plt.close()

def i_nh():

	fig = plt.figure()

	ax1 = plt.subplot2grid((3,9), (0,0),colspan=3,rowspan=3)
	ax2 = plt.subplot2grid((3,9), (0,3),colspan=3,rowspan=3)
	ax3 = plt.subplot2grid((3,9), (0,6),colspan=3,rowspan=3)

	i_h = np.empty((3,npix))

	if (model == '353'):
		fig.suptitle(r'$I_{353}/{\rm H_I}$ vs. $T_d$')
		i_h[0][:]  = I_nu_hot[0][:]/HI
		i_h[1][:]  = I_353_planck/HI
		i_h[2][:]  = I_353_gnilc/HI
		aedges = np.arange(0,1.5,0.015)
		aedges = aedges*1e-41
		bedges = np.arange(15,25,0.10)
		file       = 'I_H_Td_353'

	if (model == '545'):
		fig.suptitle(r'$I_{545}/{\rm H_I}$ vs. $T_d$')
		i_h[0][:]  = I_nu_hot[1][:]/HI
		i_h[1][:]  = I_545_planck/HI
		i_h[2][:]  = I_545_gnilc/HI
		aedges = np.arange(0,4,0.04)
		aedges = aedges*1e-21
		bedges = np.arange(10,25,0.15)
		file       = 'I_H_Td_545'

	if (model == '857'):
		fig.suptitle(r'$I_{857}/{\rm H_I}$ vs. $T_d$')
		i_h[0][:]  = I_nu_hot[2][:]/HI
		i_h[1][:]  = I_857_planck/HI
		i_h[2][:]  = I_857_gnilc/HI
		aedges = np.arange(0,4,0.04)
		aedges = aedges*1e-21
		bedges = np.arange(10,25,0.15)
		file       = 'I_H_Td_857'

	if (model == '3000'):
		fig.suptitle(r'$I_{3000}/{\rm H_I}$ vs. $T_d$')
		i_h[0][:]  = I_nu_hot[3][:]/HI
		i_h[1][:]  = I_3000_planck/HI
		i_h[2][:]  = I_3000_gnilc/HI
		aedges = np.arange(0,4,0.04)
		aedges = aedges*1e-21
		bedges = np.arange(10,25,0.15)
		file       = 'I_H_Td_3000'

	print(min(i_h[0][:]))
	print(max(i_h[0][:]))

	H1, aedges, bedges = np.histogram2d(i_h[0][:], dust_T_hot, bins=(aedges,bedges),normed=True)
	H2, aedges, bedges = np.histogram2d(i_h[1][:], Td_planck, bins=(aedges,bedges),normed=True)
	H3, aedges, bedges = np.histogram2d(i_h[2][:], Td_gnilc, bins=(aedges,bedges),normed=True)

	H1 = H1.T
	H2 = H2.T
	H3 = H3.T

	X1,Y1 = np.meshgrid(aedges, bedges)

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	# ax1.set_aspect('auto',adjustable='box-forced')
	ax1.set_title(r'Two MBBs',size=8)
	ax1.set_ylabel(r'$T_d$', size=25)
	# ax1.set_ylim([ymin,ylim])

	ax2.pcolormesh(X1,Y1,H2,cmap=plt.get_cmap('hot_r'))
	# ax2.set_aspect('auto',adjustable='box-forced')
	ax2.set_title(r'Planck 2015',size=8)

	ax3.pcolormesh(X1,Y1,H3,cmap=plt.get_cmap('hot_r'))
	# ax3.set_aspect('auto',adjustable='box-forced')
	ax3.set_title(r'GNILC',size=8)

	plt.show()

# initialize()
# i_nh()

# E(B-V) functions
# -----------------------------
def ebv_maps():

	for i in range(npix):
		if (redz[i] == -1.6375e+30):
			redz[i] = np.nan

		if (np.isnan(redz[i])):
			reddening[i]     = np.nan
			tau_ebv_nu[0][i] = np.nan
			tau_ebv_nu[1][i] = np.nan
			tau_ebv_nu[2][i] = np.nan
			tau_ebv_nu[3][i] = np.nan

		else:
			A_v_bayestar[i] = 2.742*reddening[i]

	fig, (((ax1), (ax2),(ax3)), ((ax4), (ax5), (ax6))) = plt.subplots(2,3) 

	if (model == '353'):
		fig.suptitle(r'$\tau_{353}/A_V$', size=20)
		file = 'tau_353_A_v'
		mini = -6
		maxi = -3

	if (model == '545'):
		fig.suptitle(r'$\tau_{545}/A_V$', size=20)
		file = 'tau_545_A_v'
		mini = -6
		maxi = -3

	if (model == '857'):
		fig.suptitle(r'$\tau_{857}/A_V$', size=20)
		file = 'tau_857_A_v'
		mini = -5
		maxi = -3

	if (model == '3000'):
		fig.suptitle(r'$\tau_{3000}/A_V$', size=20)
		file = 'tau_3000_A_v'
		mini = -4
		maxi = -2

	map1 = tau_ebv_nu[0][:]/A_v_bayestar
	map2 = tau_ebv_nu[1][:]/A_v_bayestar
	map3 = tau_ebv_nu[3][:]/A_v_bayestar
	map4 = tau_ebv_nu[2][:]/A_v_bayestar
	map5 = tau_ebv_nu[4][:]/A_v_bayestar
	map6 = tau_ebv_nu[5][:]/A_v_bayestar

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')
	ax5.axis('off')
	ax6.axis('off')

	plt.sca(ax1)
	hp.mollview(np.log10(map1),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'Two-Component Dust MBB')
	plt.sca(ax2)
	hp.mollview(np.log10(map2),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'Planck 2015 MBB')
	plt.sca(ax3)
	hp.mollview(np.log10(map3),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'$B_{\nu}(19.6K)$')
	plt.sca(ax4)
	hp.mollview(np.log10(map4),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'GNILC MBB')
	plt.sca(ax5)
	hp.mollview(np.log10(map5),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'M\&F 2014')
	plt.sca(ax6)
	hp.mollview(np.log10(map6),cmap=pl_cmap,hold=True,min=mini,max=maxi, title=r'\texttt{premise]')
	plt.savefig('tau/'+version+'/a_v/'+file, dpi=300,bbox_inches='tight')
	# plt.show()
	plt.close()

def ebv_scatter():

	dummyz1 = tau_ebv_nu[0][:]
	dummyz2 = tau_ebv_nu[1][:]
	dummyz3 = tau_ebv_nu[2][:]
	dummyz4 = tau_ebv_nu[3][:]
	dummyz5 = tau_ebv_nu[4][:]
	dummyz6 = tau_ebv_nu[5][:]

	for i in range(npix):
		if (redz[i] == -1.6375e+30):
			redz[i] = np.nan

		if (np.isnan(redz[i])):
			reddening[i] = np.nan
			dummyz1[i]   = np.nan
			dummyz2[i]   = np.nan
			dummyz3[i]   = np.nan
			dummyz4[i]   = np.nan
			dummyz5[i]   = np.nan
			dummyz6[i]   = np.nan

		else:
			A_v_bayestar[i] = 2.742*reddening[i]

	A_v_bayestar_new = A_v_bayestar[np.where(True != np.isnan(A_v_bayestar))]

	tau_ebv_2d  = tau_ebv_nu[0][np.where(True != np.isnan(reddening))]
	tau_ebv_pl  = tau_ebv_nu[1][np.where(True != np.isnan(reddening))]
	tau_ebv_bnu = tau_ebv_nu[2][np.where(True != np.isnan(reddening))]
	tau_ebv_gn  = tau_ebv_nu[3][np.where(True != np.isnan(reddening))]
	tau_ebv_mf  = tau_ebv_nu[4][np.where(True != np.isnan(dummyz5))]
	A_v_mf      = A_v_bayestar[np.where(True != np.isnan(dummyz5))]
	tau_ebv_pre = tau_ebv_nu[5][np.where(True != np.isnan(reddening))]

	new_2d  = []
	new_pl  = []
	new_bnu = []
	new_gn  = []
	new_mf  = []
	new_pre = []
	new_A_v = []
	new_A_v2= []

	for i in range(len(tau_ebv_2d)):
		if (tau_ebv_2d[i] < 1.0):
			new_2d.append(tau_ebv_2d[i])
			new_pl.append(tau_ebv_pl[i])
			new_bnu.append(tau_ebv_bnu[i])
			new_gn.append(tau_ebv_gn[i])
			new_pre.append(tau_ebv_pre[i])
			new_A_v.append(A_v_bayestar_new[i])

	for i in range(len(tau_ebv_mf)):
		if (tau_ebv_mf[i] < 1.0):
			new_mf.append(tau_ebv_mf[i])
			new_A_v2.append(A_v_bayestar_new[i])

	fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2,6,figsize=(12,4),sharex=True,sharey='row')

	if (model == '353'):
		fig.suptitle(r'$A_V$ vs. $\tau_{353}$', size=20)
		ymin    = 1e-7
		ylim    = 1e-5		
		yedges1 = np.logspace(-7,-5,num=100)
		file = 'A_V_scatter_353'

	if (model == '545'):
		fig.suptitle(r'$A_V$ vs. $\tau_{545}$', size=20)
		ymin    = 1e-7
		ylim    = 2e-5		
		yedges1 = np.logspace(-7,-4.5,num=100)
		file = 'A_V_scatter_545'

	if (model == '857'):
		fig.suptitle(r'$A_V$ vs. $\tau_{857}$', size=20)
		ymin    = 1e-6
		ylim    = 4e-5		
		yedges1 = np.logspace(-7,-4.3,num=100)
		file = 'A_V_scatter_857'

	if (model == '3000'):
		fig.suptitle(r'$A_V$ vs. $\tau_{3000}$', size=20)
		ymin    = 1e-5
		ylim    = 1e-3
		yedges1 = np.logspace(-5,-3,num=100)
		file = 'A_V_scatter_3000'

	fig.text(0.5, 0.02, r'$A_V$', ha='center', size=15)

	xedges1 = np.logspace(-2,1,num=100)

	# Current two-component MBB
	# -----------------------------------------------------------------------------------------
	slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(new_A_v,new_2d)

	H1, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_2d, bins=(xedges1,yedges1),normed=True)
	H1     = H1.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	mean1,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_2d,bins=xedges1,statistic='mean')
	std1, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_2d,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# Planck 2015
	# -----------------------------------------------------------------------------------------
	H2, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_pl, bins=(xedges1,yedges1),normed=True)
	H2     = H2.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(new_A_v,new_pl)

	mean2,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_pl,bins=xedges1,statistic='mean')
	std2, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_pl,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# GNILC
	# -----------------------------------------------------------------------------------------
	H3, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_gn, bins=(xedges1,yedges1),normed=True)
	H3     = H3.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(new_A_v,new_gn)

	mean3,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_gn,bins=xedges1,statistic='mean')
	std3, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_gn,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# I_Planck/B_nu(19.6K)
	# -----------------------------------------------------------------------------------------
	H4, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_bnu, bins=(xedges1,yedges1),normed=True)
	H4     = H4.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(new_A_v,new_bnu)

	mean4,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_bnu,bins=xedges1,statistic='mean')
	std4, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_bnu,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# Meisner/Finkbeiner 2014
	# -----------------------------------------------------------------------------------------
	H5, xedges1, yedges1 = np.histogram2d(A_v_mf, tau_ebv_mf, bins=(xedges1,yedges1),normed=True)
	H5     = H5.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(new_A_v2, new_mf)

	mean5,edges, nums = stats.binned_statistic(A_v_mf, tau_ebv_mf,bins=xedges1,statistic='mean')
	std5, edges, nums = stats.binned_statistic(A_v_mf, tau_ebv_mf,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------
	# premise 2019
	# -----------------------------------------------------------------------------------------
	H6, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_pre, bins=(xedges1,yedges1),normed=True)
	H6     = H6.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope6, intercept6, r_value6, p_value6, std_err6 = stats.linregress(new_A_v,new_pre)

	mean6,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_pre,bins=xedges1,statistic='mean')
	std6, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_pre,bins=xedges1,statistic='std')
	# -----------------------------------------------------------------------------------------

	centers = (edges[:-1] + edges[1:])/2

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	ax1.set_title(r'Two MBBs',size=8)
	ax1.set_ylabel(r'$\tau$',size=25)
	ax1.set_xscale('log')
	ax1.set_yscale('log')	
	ax1.set_xlim([0.03,1.0])
	ax1.set_ylim([ymin,ylim])

	ax2.pcolormesh(X1,Y1,H2,cmap=plt.get_cmap('hot_r'))
	ax2.set_title(r'Planck 2015',size=8)

	ax3.pcolormesh(X1,Y1,H3,cmap=plt.get_cmap('hot_r'))
	ax3.set_title(r'GNILC',size=8)

	ax4.pcolormesh(X1,Y1,H4,cmap=plt.get_cmap('hot_r'))
	ax4.set_title(r'$B_{\nu}(19.6K)$',size=8)

	ax5.pcolormesh(X1,Y1,H5,cmap=plt.get_cmap('hot_r'))
	ax5.set_title(r'Meisner/Finkbeiner',size=8)
	
	ax6.pcolormesh(X1,Y1,H6,cmap=plt.get_cmap('hot_r'))
	ax6.set_title(r'\texttt{premise}',size=8)

	ax7.plot(centers,std1/mean1,'k-')
	ax7.set_ylabel(r'$\sigma/\mu$', size=15)
	ax7.tick_params(axis='both',labelsize=12)
	ax7.set_xlim([0.03,1.0])
	ax7.set_ylim([0,2])

	ax8.plot(centers,std2/mean2,'k-')
	ax9.plot(centers,std3/mean3,'k-')
	ax10.plot(centers,std4/mean4,'k-')
	ax11.plot(centers,std5/mean5,'k-')
	ax12.plot(centers,std6/mean6,'k-')

	plt.savefig('tau/'+version+'/a_v/'+file, dpi=1000,bbox_inches='tight')
	# plt.show()
	plt.close()

#==========================================================================================
#==========================================================================================

def calc():
	plots = input('Which correlation? (1: HI, 2: A_v, 3: both) ')
	maps  = input('Which plots? (1: maps, 2: scatter, 3: both) ')
	print('-----------------------------')

	if (plots == 1):
		if (maps == 1):
			print('Plotting HI_maps at '+model+'GHz.')
			HI_maps()
			print('-----------------------------')
		if (maps == 2):
			print('Plotting HI_scatter at '+model+'GHz.')
			HI_scatter()
			print('-----------------------------')
		if (maps == 3):
			print('Plotting HI_maps at '+model+'GHz.')
			HI_maps()
			print('-----------------------------')
			print('Plotting HI_scatter at '+model+'GHz.')
			HI_scatter()
			print('-----------------------------')
	if (plots == 2):
		if (maps == 1):
			print('Plotting A_v maps at '+model+'GHz.')
			ebv_maps()
			print('-----------------------------')
		if (maps == 2):
			print('Plotting A_v scatter at '+model+'GHz.')
			ebv_scatter()
			print('-----------------------------')
		if (maps == 3):
			print('Plotting A_v maps at '+model+'GHz.')
			ebv_maps()
			print('-----------------------------')
			print('Plotting A_v scatter at '+model+'GHz.')
			ebv_scatter()
			print('-----------------------------')
	if (plots == 3):
		if (maps == 1):
			print('Plotting HI_maps at '+model+'GHz.')
			HI_maps()
			print('-----------------------------')
			print('Plotting A_v maps at '+model+'GHz.')
			ebv_maps()
			print('-----------------------------')
		if (maps == 2):
			print('Plotting HI_scatter at '+model+'GHz.')
			HI_scatter()
			print('-----------------------------')
			print('Plotting A_v scatter at '+model+'GHz.')
			ebv_scatter()
			print('-----------------------------')
		if (maps == 3):
			print('Plotting HI_maps at '+model+'GHz.')
			HI_maps()
			print('-----------------------------')
			print('Plotting HI_scatter at '+model+'GHz.')
			HI_scatter()
			print('-----------------------------')
			print('Plotting A_v maps at '+model+'GHz.')
			ebv_maps()
			print('-----------------------------')
			print('Plotting A_v scatter at '+model+'GHz.')
			ebv_scatter()
			print('-----------------------------')

	print('')
	global again	
	again = raw_input('Plot another frequency? (y/n) ')

while (again == 'y'):
	print('-----------------------------')
	initialize()
	calc()

print('==================================')