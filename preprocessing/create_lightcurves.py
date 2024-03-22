import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.interpolate as interpolate
import h5py
import time as timer
import sncosmo
import sys
import tqdm
sys.path.append("../")
import utils

def interp_nan(t, mag):
    ii = np.where(~np.isnan(mag))[0]
    if len(ii) > 1:
        f = interpolate.interp1d(
            t[ii], mag[ii], fill_value="extrapolate"
        )
        # Do the interpolation
        mag = f(t)
    return mag

# This is the setup for my lightcurves
X_filename = "./X_afgpy.h5"
Y_filename = "./Y_afgpy.h5"
lcdir = "../lcdir/"
parameter_names = ['n_ism', 'theta_obs', 'Eiso_c', 'theta_c', 'p_fs', 'eps_e_fs', 'eps_b_fs']

# Read the data
X, freqs, times = utils.read_X_file(X_filename)
Y = utils.read_Y_file(Y_filename)
filter_wavelengths = [1. , 545077196.36, 2.47968, 499654096.6, 999308193.3, 12.3984] # in Angstroms
wavelength = 299792458/np.array(freqs)[::-1]*10**10 # in Angstroms
times = np.array(times) / (3600 * 24) # in days

# TODO: add the option to clean the lc dir before we generate the new set of lc
start = timer.time()
for ind, parameter in tqdm.tqdm(enumerate(np.array(Y))):
    
    # Get the correct filename convention
    zipped_list = list(zip(parameter_names, parameter))
    filename_list = [f"{name}_{value}" for name, value in zipped_list]
    filename = "_".join([str(p) for p in filename_list])
    filename = lcdir + filename + ".dat"
    
    flux = X[ind][::-1] # wavelength has to increase, so we reverse the array
    m_tot = []

    # Go over all the filters and do the postprocessing
    for ifilt, filt in enumerate(utils.filters):
        if filt == "bessellv":
            cgs_flux = flux* 10**(-26)/(299792458*10**10)
            source = sncosmo.TimeSeriesSource(times, wavelength, cgs_flux)
            bandpass = sncosmo.get_bandpass(utils.filters[ifilt])
            m = source.bandmag(bandpass, "ab", times)
        else:
            lambda_filt = filter_wavelengths[ifilt]
            mJys = interpolate.interp1d(wavelength, flux, axis =1)(lambda_filt)
            Jys = mJys*10**(-3)
            m = -48.6 + -1 * np.log10(Jys / 1e23) * 2.5
        m = interp_nan(times, m)
        m_tot.append(m)
    np.savetxt(filename, np.vstack((m_tot)).T, header = " ".join(utils.filters), fmt = '%.6e')
    
end = timer.time()
print(f"Time elapsed: {end - start} seconds")