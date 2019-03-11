import numpy as np
import h5py
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as units
from scipy import stats
mpl.rcParams['text.usetex'] = True

#------------------------------------------------------------------------------------------
# Initialize maps
#------------------------------------------------------------------------------------------

I353_hot    = hp.read_map('NHI/v36/sample_1000/dust_hot_band27_k00010.fits')
I545_hot    = hp.read_map('NHI/v36/sample_1000/dust_hot_band31_k00010.fits')
I857_hot    = hp.read_map('NHI/v36/sample_1000/dust_hot_band34_k00010.fits')
I3000_hot   = hp.read_map('NHI/v36/sample_1000/dust_hot_band49_k00010.fits')

I353_cold   = hp.read_map('NHI/v36/sample_1000/dust_cold_band27_k00010.fits')
I545_cold   = hp.read_map('NHI/v36/sample_1000/dust_cold_band31_k00010.fits')
I857_cold   = hp.read_map('NHI/v36/sample_1000/dust_cold_band34_k00010.fits')
I3000_cold  = hp.read_map('NHI/v36/sample_1000/dust_cold_band49_k00010.fits')

dust_T_hot  = hp.read_map('NHI/v36/sample_1000/dust_hot_Td_c0001_k01000.fits')
dust_b_hot  = hp.read_map('NHI/v36/sample_1000/dust_hot_beta_c0001_k01000.fits')

dust_T_cold = hp.read_map('NHI/v36/sample_1000/dust_cold_Td_c0001_k01000.fits')
dust_b_cold = hp.read_map('NHI/v36/sample_1000/dust_cold_beta_c0001_k01000.fits')

HI          = hp.read_map('NHI/mom0_-90_90_0064.hpx.fits')
reddening   = hp.read_map('reddening/bayestar2017_n0064.fits')

#------------------------------------------------------------------------------------------
# Initialize values/functions
#------------------------------------------------------------------------------------------
h     = 6.626e-34     # Planck constant, J s
c     = 3.0e+8        # Speed of light, m/s
k     = 1.38e-23      # Boltzmann constant, J/K

beta  = 1.70
Td    = 19.65

def planck(fre,T):
	return ((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))))

def MBB(fre,bet,T):
	return ((fre/(545*1e9))**bet)*((2*h*fre**3)/c**2)*(1.0/(np.exp((h*fre)/(k*T))-1))	

# Computing Planck 2015 MBB values
MBB_353        = MBB(353*1e9,beta,Td)
MBB_545        = MBB(545*1e9,beta,Td)
MBB_857        = MBB(857*1e9,beta,Td)
MBB_3000       = MBB(3000*1e9,beta,Td)

npix           = hp.nside2npix(64)

# B_nu       = np.empty((len(bands),npix))
MBBs           = np.empty(4)

MBBs[0]        = MBB_353
MBBs[1]        = MBB_545
MBBs[2]        = MBB_857
MBBs[3]        = MBB_3000

#------------------------------------------------------------------------------------------
# Converting intensities (uK_cmb) to W sr^-1 m^-2 Hz^-1
#------------------------------------------------------------------------------------------

I_nu_hot        = np.empty((4,npix))
I_nu_cold       = np.empty((4,npix))

I_nu_hot[0][:]  = I353_hot*0.29008843E-20
I_nu_cold[0][:] = I353_cold*0.29008843E-20

I353_hot  = I353_hot*0.29008843E-20
I353_cold = I353_cold*0.29008843E-20

I_nu_hot[1][:]  = I545_hot*0.57287522E-21
I_nu_cold[1][:] = I545_cold*0.57287522E-21

I545_hot  = I545_hot*0.57287522E-21
I545_cold = I545_cold*0.57287522E-21

I_nu_hot[2][:]  = I857_hot*1e-17
I_nu_cold[2][:] = I857_cold*1e-17

I857_hot  = I857_hot*1e-17
I857_cold = I857_cold*1e-17

I_nu_hot[3][:]  = I3000_hot*1e-17
I_nu_cold[3][:] = I3000_cold*1e-17

I3000_hot  = I3000_hot*1e-17
I3000_cold = I3000_cold*1e-17


#------------------------------------------------------------------------------------------
# Calculate \tau_\nu
#------------------------------------------------------------------------------------------

tau_HI_nu  = np.empty((3,npix))
tau_ebv_nu = np.empty((3,npix))

model = raw_input('353, 545, 857, or 3000GHz? ')
print(model)

# Commander Model \tau_\nu

if (model == '353'):
	tau_HI_nu[0][:]  = I_nu_hot[0][:]/MBB(353*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[0][:]/MBB(353*1e9,dust_b_cold,dust_T_cold)
	tau_ebv_nu[0][:] = I_nu_hot[0][:]/MBB(353*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[0][:]/MBB(353*1e9,dust_b_cold,dust_T_cold)
	tau_HI_nu[1][:]  = (I_nu_hot[0][:] + I_nu_cold[0][:])/MBBs[0]
	tau_ebv_nu[1][:] = (I_nu_hot[0][:] + I_nu_cold[0][:])/MBBs[0]
	tau_HI_nu[2][:]  = (I_nu_hot[0][:] + I_nu_cold[0][:])/planck(353*1e9,19.6)
	tau_ebv_nu[2][:] = (I_nu_hot[0][:] + I_nu_cold[0][:])/planck(353*1e9,19.6)

if (model == '545'):
	tau_HI_nu[0][:]  = I_nu_hot[1][:]/MBB(545*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[1][:]/MBB(545*1e9,dust_b_cold,dust_T_cold)
	tau_ebv_nu[0][:] = I_nu_hot[1][:]/MBB(545*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[1][:]/MBB(545*1e9,dust_b_cold,dust_T_cold)
	tau_HI_nu[1][:]  = (I_nu_hot[1][:] + I_nu_cold[1][:])/MBBs[1]
	tau_ebv_nu[1][:] = (I_nu_hot[1][:] + I_nu_cold[1][:])/MBBs[1]
	tau_HI_nu[2][:]  = (I_nu_hot[1][:] + I_nu_cold[1][:])/planck(545*1e9,19.6)
	tau_ebv_nu[2][:] = (I_nu_hot[1][:] + I_nu_cold[1][:])/planck(545*1e9,19.6)

if (model == '857'):
	tau_HI_nu[0][:]  = I_nu_hot[2][:]/MBB(857*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[2][:]/MBB(857*1e9,dust_b_cold,dust_T_cold)
	tau_ebv_nu[0][:] = I_nu_hot[2][:]/MBB(857*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[2][:]/MBB(857*1e9,dust_b_cold,dust_T_cold)
	tau_HI_nu[1][:]  = (I_nu_hot[2][:] + I_nu_cold[2][:])/MBBs[2]
	tau_ebv_nu[1][:] = (I_nu_hot[2][:] + I_nu_cold[2][:])/MBBs[2]
	tau_HI_nu[2][:]  = (I_nu_hot[2][:] + I_nu_cold[2][:])/planck(857*1e9,19.6)
	tau_ebv_nu[2][:] = (I_nu_hot[2][:] + I_nu_cold[2][:])/planck(857*1e9,19.6)

if (model == '3000'):
	tau_HI_nu[0][:]  = I_nu_hot[3][:]/MBB(2998*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[3][:]/MBB(2998*1e9,dust_b_cold,dust_T_cold)
	tau_ebv_nu[0][:] = I_nu_hot[3][:]/MBB(2998*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[3][:]/MBB(2998*1e9,dust_b_cold,dust_T_cold)
	tau_HI_nu[1][:]  = (I_nu_hot[3][:] + I_nu_cold[3][:])/MBBs[3]
	tau_ebv_nu[1][:] = (I_nu_hot[3][:] + I_nu_cold[3][:])/MBBs[3]
	tau_HI_nu[2][:]  = (I_nu_hot[3][:] + I_nu_cold[3][:])/planck(2998*1e9,19.6)
	tau_ebv_nu[2][:] = (I_nu_hot[3][:] + I_nu_cold[3][:])/planck(2998*1e9,19.6)

#------------------------------------------------------------------------------------------
# HI and E(B-V) plotting
#------------------------------------------------------------------------------------------

# HI functions
# -----------------------------
HI_masked  = HI

def HI_maps():

	fig, ((ax1), (ax2), (ax3)) = plt.subplots(3,1)

	if (model == '353'):
		fig.suptitle(r'$\tau_{353}$/N ${\rm H_I}$', size=20)

	if (model == '545'):
		fig.suptitle(r'$\tau_{545}$/N ${\rm H_I}$', size=20)

	if (model == '857'):
		fig.suptitle(r'$\tau_{857}$/N ${\rm H_I}$', size=20)

	if (model == '3000'):
		fig.suptitle(r'$\tau_{3000}$/N ${\rm H_I}$', size=20)

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	hp.mollview(tau_HI_nu[0][:]/HI,norm='log',sub=(1,3,1),min=1e-24,max=1e-21, title=r'Two-Component Dust MBB')
	hp.mollview(tau_HI_nu[1][:]/HI,norm='log',sub=(1,3,2),min=1e-24,max=1e-21, title=r'Planck 2013 MBB')
	hp.mollview(tau_HI_nu[2][:]/HI,norm='log',sub=(1,3,3),min=1e-24,max=1e-21, title=r'$B_{\nu}(19.6K)$')
	plt.show()

def HI_scatter():

	for i in range(npix):
		if (HI[i] > 4e20):
			HI_masked[i]  = np.nan

	for i in range(npix):
		if (np.isnan(HI_masked[i])):
			tau_HI_nu[0][i] = np.nan
			tau_HI_nu[1][i] = np.nan
			tau_HI_nu[2][i] = np.nan
			
	tau_HI_2d  = tau_HI_nu[0][np.where(True != np.isnan(HI_masked))]
	tau_HI_pl  = tau_HI_nu[1][np.where(True != np.isnan(HI_masked))]
	tau_HI_bnu = tau_HI_nu[2][np.where(True != np.isnan(HI_masked))]
	HI_new     = HI_masked[np.where(True != np.isnan(HI_masked))]

	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, sharex=True, sharey=True)

	if (model == '353'):
		fig.suptitle(r'HI Column Density vs. $\tau_{353}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1*1e20
		yedges1 = np.arange(0,1.2,0.012)
		yedges1 = yedges1*1e-2

	if (model == '545'):
		fig.suptitle(r'HI Column Density vs. $\tau_{545}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1*1e20
		yedges1 = np.arange(0,1.5,0.015)
		yedges1 = yedges1*1e-2

	if (model == '857'):
		fig.suptitle(r'HI Column Density vs. $\tau_{857}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1*1e20
		yedges1 = np.arange(0,25,0.025)
		yedges1 = yedges1*1e-2

	if (model == '3000'):
		fig.suptitle(r'HI Column Density vs. $\tau_{3000}$', size=20)
		xedges1 = np.arange(0,4,0.04)
		xedges1 = xedges1*1e20
		yedges1 = np.arange(0,25,0.025)
		yedges1 = yedges1*1e-2	

	fig.text(0.5, 0.05, r'NH ${\rm [cm^{-2}]}$', ha='center', size=15)

	H1, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_2d, bins=(xedges1,yedges1),normed=True)
	H1     = H1.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(HI_new,tau_HI_2d)

	mean1,edges, nums = stats.binned_statistic(HI_new,tau_HI_2d,bins=xedges1,statistic='mean')
	std1, edges, nums = stats.binned_statistic(HI_new,tau_HI_2d,bins=xedges1,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	ax1.plot(xedges1,slope1*xedges1+intercept1, color='black', linestyle='--')
	ax1.set_title(r'Two-Component Dust MBB',size=20)
	ax1.tick_params(labelsize=14)
	ax1.set_ylabel(r'$\tau$', size=25)
	ax1.set_xlim([0.5e20,3.75e20])
	ax1.set_ylim([0,0.012])

	ax4.plot(centers,std1,'k-')
	ax4.tick_params(axis='both',labelsize=14)
	ax4.set_xlim([0.5e20,3.75e20])
	ax4.set_ylim([0,1.2e-2])

	H2, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_pl, bins=(xedges1,yedges1),normed=True)
	H2     = H2.T

	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(HI_new,tau_HI_pl)

	mean2,edges, nums = stats.binned_statistic(HI_new,tau_HI_pl,bins=xedges1,statistic='mean')
	std2, edges, nums = stats.binned_statistic(HI_new,tau_HI_pl,bins=xedges1,statistic='std')

	ax2.pcolormesh(X1,Y1,H2,cmap=plt.get_cmap('hot_r'))
	ax2.plot(xedges1,slope2*xedges1+intercept2, color='black', linestyle='--')
	ax2.set_title(r'Planck 2013 MBB',size=20)
	ax2.tick_params(labelsize=14)

	ax5.plot(centers,std2,'k-')
	ax5.tick_params(axis='both',labelsize=14)

	H3, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_bnu, bins=(xedges1,yedges1),normed=True)
	H3     = H3.T

	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(HI_new,tau_HI_bnu)

	mean3,edges, nums = stats.binned_statistic(HI_new,tau_HI_bnu,bins=xedges1,statistic='mean')
	std3, edges, nums = stats.binned_statistic(HI_new,tau_HI_bnu,bins=xedges1,statistic='std')

	ax3.pcolormesh(X1,Y1,H3,cmap=plt.get_cmap('hot_r'))
	ax3.plot(xedges1,slope3*xedges1+intercept3, color='black', linestyle='--')
	ax3.set_title(r'$B_{\nu}(19.6K)$',size=20)
	ax3.tick_params(labelsize=14)

	ax6.plot(centers,std3,'k-')
	ax6.tick_params(axis='both',labelsize=14)

	plt.show()

# E(B-V) functions
# -----------------------------
A_v_bayestar    = np.empty(npix,dtype='f8')
A_v_bayestar[:] = np.nan

def ebv_maps():

	for i in range(npix):
		if (reddening[i] == -1.6375e+30):
			reddening[i] = np.nan

		if (np.isnan(reddening[i])):
			tau_ebv_nu[0][i] = np.nan
			tau_ebv_nu[1][i] = np.nan
			tau_ebv_nu[2][i] = np.nan

		else:
			A_v_bayestar[i] = 2.742*reddening[i]

	fig, ((ax1), (ax2), (ax3)) = plt.subplots(3,1)

	if (model == '353'):
		fig.suptitle(r'$\tau_{353}/A_V$', size=20)

	if (model == '545'):
		fig.suptitle(r'$\tau_{545}/A_V$', size=20)

	if (model == '857'):
		fig.suptitle(r'$\tau_{857}/A_V$', size=20)

	if (model == '3000'):
		fig.suptitle(r'$\tau_{3000}/A_V$', size=20)

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	hp.mollview(tau_ebv_nu[0][:]/HI,norm='log',sub=(1,3,1),min=1e-24,max=1e-21, title=r'Two-Component Dust MBB')
	hp.mollview(tau_ebv_nu[1][:]/HI,norm='log',sub=(1,3,2),min=1e-24,max=1e-21, title=r'Planck 2013 MBB')
	hp.mollview(tau_ebv_nu[2][:]/HI,norm='log',sub=(1,3,3),min=1e-24,max=1e-21, title=r'$B_{\nu}(19.6K)$')
	plt.show()

def ebv_scatter():

	for i in range(npix):
		if (reddening[i] == -1.6375e+30):
			reddening[i] = np.nan

		if (np.isnan(reddening[i])):
			tau_ebv_nu[0][i] = np.nan
			tau_ebv_nu[1][i] = np.nan
			tau_ebv_nu[2][i] = np.nan

		else:
			A_v_bayestar[i] = 2.742*reddening[i]

	A_v_bayestar_new = A_v_bayestar[np.where(True != np.isnan(A_v_bayestar))]

	tau_ebv_2d  = tau_ebv_nu[0][np.where(True != np.isnan(reddening))]
	tau_ebv_pl  = tau_ebv_nu[1][np.where(True != np.isnan(reddening))]
	tau_ebv_bnu = tau_ebv_nu[2][np.where(True != np.isnan(reddening))]

	new_2d  = []
	new_pl  = []
	new_bnu = []
	new_A_v = []

	for i in range(len(tau_ebv_2d)):
		if (tau_ebv_2d[i] < 1.0):
			new_2d.append(tau_ebv_2d[i])
			new_pl.append(tau_ebv_pl[i])
			new_bnu.append(tau_ebv_bnu[i])
			new_A_v.append(A_v_bayestar_new[i])

	fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, sharex=True, sharey=True)

	if (model == '353'):
		fig.suptitle(r'HI Column Density vs. $\tau_{353}$', size=20)

	if (model == '545'):
		fig.suptitle(r'HI Column Density vs. $\tau_{545}$', size=20)

	if (model == '857'):
		fig.suptitle(r'HI Column Density vs. $\tau_{857}$', size=20)

	if (model == '3000'):
		fig.suptitle(r'HI Column Density vs. $\tau_{3000}$', size=20)

	fig.text(0.5, 0.05, r'$A_V$', ha='center', size=20)

	xedges1 = np.logspace(-2,1,num=100)
	yedges1 = np.logspace(-4,0,num=100)

	slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(new_A_v,new_2d)

	H1, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_2d, bins=(xedges1,yedges1),normed=True)
	H1     = H1.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	mean1,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_2d,bins=xedges1,statistic='mean')
	std1, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_2d,bins=xedges1,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	ax1.plot(xedges1,slope1*xedges1+intercept1, color='black', linestyle='--')
	ax1.set_title(r'Two-Component Dust MBB',size=20)
	ax1.set_ylabel(r'$\tau$',size=25)
	ax1.tick_params(labelsize=14)
	ax1.set_xscale('log')
	ax1.set_yscale('log')	
	ax1.set_xlim([0.03,1.0])
	ax1.set_ylim([1e-4,0.1])

	ax4.plot(centers,std1,'k-')
	ax4.tick_params(axis='both',labelsize=14)
	ax4.set_ylabel('Standard Deviation', size=15)
	ax4.set_xlim([0.03,1.0])
	ax4.set_ylim([1e-4,0.1])


	H2, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_pl, bins=(xedges1,yedges1),normed=True)
	H2     = H2.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(new_A_v,new_pl)

	mean2,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_pl,bins=xedges1,statistic='mean')
	std2, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_pl,bins=xedges1,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax2.pcolormesh(X1,Y1,H2,cmap=plt.get_cmap('hot_r'))
	ax2.plot(xedges1,slope2*xedges1+intercept2, color='black', linestyle='--')
	ax2.set_title(r'Planck 2013 MBB',size=20)
	ax2.tick_params(labelsize=14)

	ax5.plot(centers,std2,'k-')
	ax5.tick_params(axis='both',labelsize=14)


	H3, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_bnu, bins=(xedges1,yedges1),normed=True)
	H3     = H3.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(new_A_v,new_bnu)

	mean3,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_bnu,bins=xedges1,statistic='mean')
	std3, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_bnu,bins=xedges1,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax3.pcolormesh(X1,Y1,H3,cmap=plt.get_cmap('hot_r'))
	ax3.plot(xedges1,slope3*xedges1+intercept3, color='black', linestyle='--')
	ax3.set_title(r'$B_{\nu}(19.6K)$',size=20)
	ax3.tick_params(labelsize=14)

	ax6.plot(centers,std3,'k-')
	ax6.tick_params(axis='both',labelsize=14)

	plt.show()

# HI_maps()
# HI_scatter()
# ebv_maps()
ebv_scatter()