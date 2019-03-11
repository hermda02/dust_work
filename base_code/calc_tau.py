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

CIB_mask    = hp.read_map('CIB/CIB_mask_inverted_n0064.fits')
HI          = hp.read_map('NHI/mom0_-90_90_0064.hpx.fits')
HI4PI       = hp.read_map('/home/daniel/Data/NHI_HPX.fits')
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

bands          = np.empty(4)
string_band    = np.empty(4)

MBBs           = np.empty(len(bands))

MBBs[0]        = MBB_353
MBBs[1]        = MBB_545
MBBs[2]        = MBB_857
MBBs[3]        = MBB_3000

bands[0]       = 353
bands[1]       = 545
bands[2]       = 857
bands[3]       = 3000

string_band[0] = '353'
string_band[1] = '545'
string_band[2] = '857'
string_band[3] = '3000'

#------------------------------------------------------------------------------------------
# Converting intensities (uK_cmb) to W sr^-1 m^-2 Hz^-1
#------------------------------------------------------------------------------------------

I_nu_hot        = np.empty((len(bands),npix))
I_nu_cold       = np.empty((len(bands),npix))

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

tau_HI_nu  = np.empty((len(bands),npix))
tau_ebv_nu = np.empty((len(bands),npix))

model = raw_input('two-component dust (2d), 2013 Planck results (2013), B_nu(19.6K) (bnu), or compare? ')
print(model)

# Commander Model \tau_\nu

if (model == '2d'):
	for i in range(len(bands)):
		tau_HI_nu[i][:]  = I_nu_hot[i][:]/MBB(bands[i]*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[i][:]/MBB(bands[i]*1e9,dust_b_cold,dust_T_cold)
		tau_ebv_nu[i][:] = I_nu_hot[i][:]/MBB(bands[i]*1e9,dust_b_hot,dust_T_hot) + I_nu_cold[i][:]/MBB(bands[i]*1e9,dust_b_cold,dust_T_cold)

if ((model == 'bnu')):
	for i in range(len(bands)):
		tau_HI_nu[i][:]  = I_nu_hot[i][:]/planck(bands[i]*1e9,19.6) + I_nu_cold[i][:]/planck(bands[i]*1e9,19.6)
		tau_ebv_nu[i][:] = I_nu_hot[i][:]/planck(bands[i]*1e9,19.6) + I_nu_cold[i][:]/planck(bands[i]*1e9,19.6)

if (model == '2013'):
	for i in range(len(bands)):
		tau_HI_nu[i][:]  = (I_nu_hot[i][:] + I_nu_cold[i][:])/MBBs[i]
		tau_ebv_nu[i][:] = (I_nu_hot[i][:] + I_nu_cold[i][:])/MBBs[i]

#------------------------------------------------------------------------------------------
# HI and E(B-V) plotting
#------------------------------------------------------------------------------------------

# HI functions
# -----------------------------
HI_masked  = HI

def HI_maps():

	fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2)

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')
	hp.mollview(tau_ebv_nu[0][:]/HI,norm='log',sub=(2,2,1),min=1e-24,max=1e-21, title=r'$\tau_{353}/N_{\rm H}$')
	hp.mollview(tau_ebv_nu[1][:]/HI,norm='log',sub=(2,2,2),min=1e-24,max=1e-21, title=r'$\tau_{545}/N_{\rm H}$')
	hp.mollview(tau_ebv_nu[2][:]/HI,norm='log',sub=(2,2,3),min=1e-24,max=1e-21, title=r'$\tau_{857}/N_{\rm H}$')
	hp.mollview(tau_ebv_nu[3][:]/HI,norm='log',sub=(2,2,4),min=1e-24,max=1e-21, title=r'$\tau_{3000}/N_{\rm H}$')
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
			tau_HI_nu[3][i] = np.nan
			
	tau_HI_353  = tau_HI_nu[0][np.where(True != np.isnan(HI_masked))]
	tau_HI_545  = tau_HI_nu[1][np.where(True != np.isnan(HI_masked))]
	tau_HI_857  = tau_HI_nu[2][np.where(True != np.isnan(HI_masked))]
	tau_HI_3000 = tau_HI_nu[3][np.where(True != np.isnan(HI_masked))]
	HI_new      = HI_masked[np.where(True != np.isnan(HI_masked))]

	fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, sharex=True, sharey=True)

	if (model == '2d'):
		fig.suptitle(r'HI Column Density vs. $\tau_{\nu,hot} + \tau_{\nu,cold}$',size=20)
	if (model == '2013'):
		fig.suptitle(r'HI Column Density vs. $\tau_{\nu}$ (Planck 2013)',size=20)
	if (model == 'bnu'):
		fig.suptitle(r'HI Column Density vs. $I_{\nu}/B_{\nu}(19.6K)$',size=20)
	fig.text(0.5, 0.05, r'NH ${\rm [cm^{-2}]}$', ha='center', size=15)

	# 353GHz
	xedges1 = np.arange(0,4,0.04)
	xedges1 = xedges1*1e20
	yedges1 = np.arange(0,1.2,0.012)
	yedges1 = yedges1*1e-2

	H1, xedges1, yedges1 = np.histogram2d(HI_new,tau_HI_353, bins=(xedges1,yedges1),normed=True)
	H1     = H1.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	mean1,edges, nums = stats.binned_statistic(HI_new,tau_HI_353,bins=xedges1,statistic='mean')
	std1, edges, nums = stats.binned_statistic(HI_new,tau_HI_353,bins=xedges1,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	ax1.set_title(r'$353$ GHz',size=20)
	ax1.tick_params(labelsize=14)
	ax1.set_ylabel(r'$\tau$', size=25)
	ax1.set_xlim([0.5e20,3.75e20])
	ax1.set_ylim([0,0.012])

	ax5.plot(centers,std1,'k-')
	ax5.tick_params(axis='both',labelsize=14)
	ax5.set_ylabel('Standard Deviation', size=15)
	ax5.set_xlim([0.5e20,3.75e20])
	ax5.set_ylim([0,1.2e-2])

	# 545GHz
	xedges2 = np.arange(0,4,0.04)
	xedges2 = xedges2*1e20
	yedges2 = np.arange(0,1.5,0.015)
	yedges2 = yedges2*1e-2

	H2, xedges2, yedges2 = np.histogram2d(HI_new,tau_HI_545, bins=(xedges2,yedges2),normed=True)
	H2     = H2.T
	X2, Y2 = np.meshgrid(xedges2,yedges2)

	mean2,edges, nums = stats.binned_statistic(HI_new,tau_HI_545,bins=xedges2,statistic='mean')
	std2, edges, nums = stats.binned_statistic(HI_new,tau_HI_545,bins=xedges2,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax2.pcolormesh(X2,Y2,H2,cmap=plt.get_cmap('hot_r'))
	ax2.set_title(r'$545$ GHz',size=20)
	ax2.set_xlim([0.5e20,3.75e20])
	ax2.set_ylim([0,0.012])
	ax6.plot(centers,std2,'k-')
	ax6.tick_params(labelsize=14)
	ax6.set_xlim([0.5e20,3.75e20])
	ax6.set_ylim([0,1.2e-2])

	# 857GHz
	xedges3 = np.arange(0,4,0.04)
	xedges3 = xedges3*1e20
	yedges3 = np.arange(0,25,0.025)
	yedges3 = yedges3*1e-2

	H3, xedges3, yedges3 = np.histogram2d(HI_new,tau_HI_857, bins=(xedges3,yedges3),normed=True)
	H3     = H3.T
	X3, Y3 = np.meshgrid(xedges3,yedges3)

	mean3,edges, nums = stats.binned_statistic(HI_new,tau_HI_857,bins=xedges3,statistic='mean')
	std3, edges, nums = stats.binned_statistic(HI_new,tau_HI_857,bins=xedges3,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax3.pcolormesh(X3,Y3,H3,cmap=plt.get_cmap('hot_r'))
	ax3.set_title(r'$857$ GHz',size=20)
	ax3.set_xlim([0.5e20,3.75e20])
	ax3.set_ylim([0,1.2e-2])

	ax7.plot(centers,std3,'k-')
	ax7.tick_params(labelsize=14)
	ax7.set_xlim([0.5e20,3.75e20])
	ax7.set_ylim([0,1.2e-2])
	
	# 3000 GHz	
	xedges4 = np.arange(0,4,0.04)
	xedges4 = xedges4*1e20
	yedges4 = np.arange(0,25,0.025)
	yedges4 = yedges4*1e-2

	H4, xedges4, yedges4 = np.histogram2d(HI_new,tau_HI_3000, bins=(xedges4,yedges4),normed=True)
	H4     = H4.T
	X4, Y4 = np.meshgrid(xedges4,yedges4)

	mean4,edges, nums = stats.binned_statistic(HI_new,tau_HI_3000,bins=xedges4,statistic='mean')
	std4, edges, nums = stats.binned_statistic(HI_new,tau_HI_3000,bins=xedges4,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax4.pcolormesh(X4,Y4,H4,cmap=plt.get_cmap('hot_r'))
	ax4.set_title(r'$3000$ GHz',size=20)
	ax4.set_xlim([0.5e20,3.75e20])
	ax4.set_ylim([0,1.2e-2])

	ax8.plot(centers,std4,'k-')
	ax8.tick_params(labelsize=14)
	ax8.set_xlim([0.5e20,3.75e20])
	ax8.set_ylim([0,1.2e-2])

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
			tau_ebv_nu[3][i] = np.nan

		else:
			A_v_bayestar[i] = 2.742*reddening[i]


	fig, ((ax1,ax2), (ax3, ax4)) = plt.subplots(2,2)

	ax1.axis('off')
	ax2.axis('off')
	ax3.axis('off')
	ax4.axis('off')
	hp.mollview(tau_ebv_nu[0][:]/A_v_bayestar,norm='log',sub=(2,2,1),min=0.005,max=2,title=r'$\tau_{353}/A_V$')
	hp.mollview(tau_ebv_nu[1][:]/A_v_bayestar,norm='log',sub=(2,2,2),min=0.005,max=2,title=r'$\tau_{545}/A_V$')
	hp.mollview(tau_ebv_nu[2][:]/A_v_bayestar,norm='log',sub=(2,2,3),min=0.005,max=2,title=r'$\tau_{857}/A_V$')
	hp.mollview(tau_ebv_nu[3][:]/A_v_bayestar,norm='log',sub=(2,2,4),min=0.005,max=2,title=r'$\tau_{3000}/A_V$')
	plt.show()

def ebv_scatter():

	for i in range(npix):
		if (reddening[i] == -1.6375e+30):
			reddening[i] = np.nan

		if (np.isnan(reddening[i])):
			tau_ebv_nu[0][i] = np.nan
			tau_ebv_nu[1][i] = np.nan
			tau_ebv_nu[2][i] = np.nan
			tau_ebv_nu[3][i] = np.nan

		else:
			A_v_bayestar[i] = 2.742*reddening[i]

	A_v_bayestar_new = A_v_bayestar[np.where(True != np.isnan(A_v_bayestar))]

	tau_ebv_353  = tau_ebv_nu[0][np.where(True != np.isnan(reddening))]
	tau_ebv_545  = tau_ebv_nu[1][np.where(True != np.isnan(reddening))]
	tau_ebv_857  = tau_ebv_nu[2][np.where(True != np.isnan(reddening))]
	tau_ebv_3000 = tau_ebv_nu[3][np.where(True != np.isnan(reddening))]

	fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, sharex=True, sharey=True)

	if (model == '2d'):
		fig.suptitle(r'HI Column Density vs. $\tau_{\nu,hot} + \tau_{\nu,cold}$',size=20)
	if (model == '2013'):
		fig.suptitle(r'Reddening vs. $\tau_{\nu}$ (Planck 2013)',size=20)
	if (model == 'bnu'):
		fig.suptitle(r'Reddening vs. $I_{\nu}/B_{\nu}(19.6K)$',size=20)
	
	fig.text(0.5, 0.05, r'$A_V$', ha='center', size=20)

	# 353GHz
	xedges1 = np.logspace(-2,1,num=100)
	yedges1 = np.logspace(-4,0,num=100)

	H1, xedges1, yedges1 = np.histogram2d(A_v_bayestar_new, tau_ebv_353, bins=(xedges1,yedges1),normed=True)
	H1     = H1.T
	X1, Y1 = np.meshgrid(xedges1,yedges1)

	mean1,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_353,bins=xedges1,statistic='mean')
	std1, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_353,bins=xedges1,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax1.pcolormesh(X1,Y1,H1,cmap=plt.get_cmap('hot_r'))
	ax1.set_title(r'$353$ GHz',size=20)
	ax1.set_ylabel(r'$\tau$',size=25)
	ax1.tick_params(labelsize=14)
	ax1.set_xscale('log')
	ax1.set_yscale('log')	
	ax1.set_xlim([0.03,1.0])
	ax1.set_ylim([1e-4,0.1])

	ax5.plot(centers,std1,'k-')
	ax5.tick_params(axis='both',labelsize=14)
	ax5.set_ylabel('Standard Deviation', size=15)
	ax5.set_xlim([0.03,1.0])
	ax5.set_ylim([1e-4,0.1])


	# 545GHz
	xedges2 = np.logspace(-2,1,num=100)
	yedges2 = np.logspace(-4,0,num=100)

	H2, xedges2, yedges2 = np.histogram2d(A_v_bayestar_new, tau_ebv_545, bins=(xedges2,yedges2),normed=True)
	H2     = H2.T
	X2, Y2 = np.meshgrid(xedges2,yedges2)

	mean2,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_545,bins=xedges2,statistic='mean')
	std2, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_545,bins=xedges2,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax2.pcolormesh(X2,Y2,H2,cmap=plt.get_cmap('hot_r'))
	ax2.set_title(r'$545$ GHz',size=20)
	ax2.tick_params(labelsize=14)
	ax2.set_xscale('log')
	ax2.set_yscale('log')	
	ax2.set_xlim([0.03,1.0])
	ax2.set_ylim([1e-4,0.1])

	ax6.plot(centers,std2,'k-')
	ax6.tick_params(axis='both',labelsize=14)
	ax6.set_xlim([0.03,1.0])
	ax6.set_ylim([1e-4,0.1])


	# 857GHz
	xedges3 = np.logspace(-2,1,num=100)
	yedges3 = np.logspace(-4,0,num=100)

	H3, xedges3, yedges3 = np.histogram2d(A_v_bayestar_new, tau_ebv_857, bins=(xedges3,yedges3),normed=True)
	H3     = H3.T
	X3, Y3 = np.meshgrid(xedges3,yedges3)

	mean3,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_857,bins=xedges3,statistic='mean')
	std3, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_857,bins=xedges3,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax3.pcolormesh(X3,Y3,H3,cmap=plt.get_cmap('hot_r'))
	ax3.set_title(r'$857$ GHz',size=20)
	ax3.tick_params(labelsize=14)
	ax3.set_xscale('log')
	ax3.set_yscale('log')	
	ax3.set_xlim([0.01,1.0])
	ax3.set_ylim([1e-4,0.1])

	ax7.plot(centers,std3,'k-')
	ax7.tick_params(axis='both',labelsize=14)
	ax7.set_xlim([0.01,1.0])
	ax7.set_ylim([1e-4,0.1])


	# 100um
	xedges4 = np.logspace(-2,1,num=100)
	yedges4 = np.logspace(-4,0,num=100)

	H4, xedges4, yedges4 = np.histogram2d(A_v_bayestar_new, tau_ebv_3000, bins=(xedges4,yedges4),normed=True)
	H4     = H4.T
	X4, Y4 = np.meshgrid(xedges4,yedges4)

	mean4,edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_3000,bins=xedges4,statistic='mean')
	std4, edges, nums = stats.binned_statistic(A_v_bayestar_new,tau_ebv_3000,bins=xedges4,statistic='std')

	centers = (edges[:-1] + edges[1:])/2

	ax4.pcolormesh(X4,Y4,H4,cmap=plt.get_cmap('hot_r'))
	ax4.set_title(r'$3000$ GHz',size=20)
	ax4.tick_params(labelsize=14)
	ax4.set_xscale('log')
	ax4.set_yscale('log')	
	ax4.set_xlim([0.01,1.0])
	ax4.set_ylim([1e-4,0.1])

	ax8.plot(centers,std4,'k-')
	ax8.tick_params(axis='both',labelsize=14)
	ax8.set_xlim([0.01,1.0])
	ax8.set_ylim([1e-4,0.1])

	plt.show()

#------------------------------------------------------------------------------------------
# Execution
#------------------------------------------------------------------------------------------

# HI_maps()
# HI_scatter()
# ebv_maps()
ebv_scatter()
exit()