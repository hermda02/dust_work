import numpy as np
import healpy as hp
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats as stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
res_dpi = 300
ext = 'png'

# Physical constants
c = 2.99792458e10       # Speed of light, cm/s
h = 6.62606957e-27      # Planck constant, erg s
k = 1.3806488e-16       # Boltzmann constant, erg/K
mp = 1.6726218e-24      # Proton mass, g

def B_nu (nu, T):
    np.seterr(over='ignore')
    return 2.0*h*(nu**3)/(c**2*(np.expm1(h*nu/(k*T))))

def linear_fit(x, m, b):
    return m*x+b

nhi = hp.read_map('/Users/bhensley/Dropbox/DustPol/data/nhi_128_80m.fits',verbose=False)
ebv = 0.884*hp.read_map('ebv_sfd_128_80m.fits',verbose=False)
tsfd = hp.read_map('/Users/bhensley/Dropbox/DustPol/data/dust_temperature_sfd.hpx.fits',verbose=False)
#tsfd[np.isnan(tsfd)] = 17.9
#tsfd = hp.ud_grade(hp.smoothing(tsfd,fwhm=(80./60.)*np.pi/180.),128)
#hp.write_map('t_sfd.fits',tsfd,overwrite=True)
tsfd = hp.read_map('t_sfd.fits',verbose=False)
(i353,q353,u353) = hp.read_map("/Users/bhensley/Dropbox/DustPol/data/gnilc2018_dust_353_128_80m.fits",
                                   field=(0,1,2), verbose=False)
p353 = 287.5*np.sqrt(q353**2+u353**2)
#thi = hp.ud_grade(hp.read_map('/Users/bhensley/Dropbox/DustPol/data/tHI_gradient_alt_HI4PI_vels70_to_116_smooth80.fits',
#                                  verbose=False), 128)
thi = hp.ud_grade(hp.read_map('/Users/bhensley/Dropbox/DustModels/tHI_gradient_HI4PI_vels70_to_116_smooth80.fits',
                                  verbose=False), 128)

print(np.median(tsfd))
p353 = p353*B_nu(353.e9,20.)/B_nu(353.e9,tsfd)

idx = np.where((nhi < 4.e20) & (thi > 0.))

fig, ax = plt.subplots(1,1,figsize=(3.5, 3))
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'$P/N_{\rm H}\ [{\rm MJy/sr\ {\rm cm}^2}]$',fontsize=10)
ax.set_ylabel(r'$E(B-V)_{\rm SFD}/N_{\rm H}\ [{\rm mag}\ {\rm cm}^2]$',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.axis([0, 1.e-22, 0., 2.e-22])
hist, xbins, ybins, im = ax.hist2d((p353/nhi)[idx], (ebv/nhi)[idx],
                    range=[[0.,1.e-22],[0.,2.e-22]], bins=100,cmap=plt.get_cmap('hot_r'))

medians, edges, nums = stats.binned_statistic((p353/nhi)[idx],(ebv/nhi)[idx],bins=xbins,statistic='median')
centers = (edges[:-1] + edges[1:]) / 2.
ax.plot(centers, medians,'k-')

#popt, pcov = curve_fit(linear_fit, nhi[nhi<4.e20], i857[nhi<4.e20])
#cib_857 = popt[1]
#xvec = np.linspace(0.,5.e20,1000)
#ax.plot(xvec, linear_fit(xvec, *popt), 'g--')
plt.savefig('ebvnh_p.' + ext,format=ext,dpi=res_dpi,bbox_inches='tight')
plt.close('all')

dust_p = hp.read_map('/Users/bhensley/Dropbox/DustPol/data/dust_p_128_80arcmin.fits',verbose=False)
fig, ax = plt.subplots(1,1,figsize=(3.5, 3))
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel(r'$t_{\rm HI}$',fontsize=10)
ax.set_ylabel(r'$p_{353}$',fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.axis([0, 0.3, 0., 0.25])

idx = np.where(nhi < 4.e20)
hist, xbins, ybins, im = ax.hist2d(thi[idx], dust_p[idx],
                    range=[[0.,0.3],[0.,0.25]], bins=100,cmap=plt.get_cmap('hot_r'))

medians, edges, nums = stats.binned_statistic(thi[idx],dust_p[idx],bins=xbins,statistic='median')
centers = (edges[:-1] + edges[1:]) / 2.
ax.plot(centers, medians,'k-')
plt.savefig('thi_p.' + ext,format=ext,dpi=res_dpi,bbox_inches='tight')
plt.close('all')

hp.mollview(thi)
plt.savefig('thi_map.' + ext,format=ext,dpi=res_dpi,bbox_inches='tight')
plt.close('all')

hp.mollview(dust_p)
plt.savefig('p_map.' + ext,format=ext,dpi=res_dpi,bbox_inches='tight')
plt.close('all')
