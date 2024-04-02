import copy

import afterglowpy as grb
import h5py, os, numpy as np, time
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from itertools import product
from tqdm import tqdm

pc = 3.0857e18 # cm # parsec

class RunAfterglowPy():
    Z = {'jetType':     grb.jet.TopHat,     # Top-Hat jet
         'specType':    0,                  # Basic Synchrotron Spectrum
         'thetaObs':    0.0,   # Viewing angle in radians
         'E0':          1.0e52, # Isotropic-equivalent energy in erg
         'thetaCore':   0.2,    # Half-opening angle in radians
         #'thetaWing':   0.2,   # Wing angle in radians
         'n0':          1e-3,    # circumburst density in cm^{-3}
         'p':           2.2,    # electron energy distribution index
         'epsilon_e':   0.1,    # epsilon_e
         'epsilon_B':   0.01,   # epsilon_B
         'xi_N':        1.0,    # Fraction of electrons accelerated
         'd_L':         3.09e26, # Luminosity distance in cm
         'z':           0.0099}   # redshift
    mapping = {"theta_obs":"thetaObs",
               "Eiso_c":"E0",
               "n_ism":"n0",
               "theta_c":"thetaCore",
               "theta_w":"thetaWing",
               "p_fs":"p",
               "eps_e_fs":"epsilon_e",
               "eps_b_fs":"epsilon_B",
               "d_l":"d_L",
               "z":"z"
               }
    times = np.logspace(2, 3, 256) * (3600 * 24)
    freqs = np.logspace(9, 22, 256)
    times_, freqs_ = np.meshgrid(times, freqs, indexing='ij')
    def __init__(self, par_list: list):
        self.par_list = par_list

    def __call__(self, idx):
        pars = copy.deepcopy(self.par_list[idx])
        Z = copy.deepcopy(self.Z)
        for key, val in pars.items():
            if key in self.mapping.keys():
                Z[self.mapping[key]] = val
        Fnu = grb.fluxDensity(self.times_.flatten(), self.freqs_.flatten(), **Z)
        Fnu = np.reshape(Fnu, newshape=self.times_.shape)
        vals = np.array([pars[key] for key in pars.keys()])
        return (vals, Fnu)




if __name__ == '__main__':
    
    pars = {
        "n_ism": np.asarray([1.0, 0.1, 0.01, 0.001]),
        "theta_obs": np.array([0., 15., 45.0, 60., 75., 90.]) * np.pi / 180.0,
        "Eiso_c": np.asarray([1.e50, 1.e51, 1.e52, 1.e53]),
        "theta_c": np.array([5., 10., 15., 20.]) * np.pi / 180.,
        "p_fs": [2.2, 2.4, 2.6, 2.8],
        "eps_e_fs": [0.5, 0.1, 0.01, 0.001],
        "eps_b_fs": [0.5, 0.1, 0.01, 0.001],
    }

    ranges = [pars[key] for key in pars.keys()]
    result = []
    all_combinations = product(*ranges)

    grid_points = []

    for combination in all_combinations:
        new_entry = {par:val for par,val in zip(pars.keys(),combination)}
        grid_points.append(new_entry)

    afgpy = RunAfterglowPy(grid_points) #feed all the grid points into the run afterglowpy class

    #start the main loop
    start_time = time.perf_counter()
    
    parameters = []
  

    processes = os.cpu_count()
    pool = Pool(processes=5)
    jobs = [pool.apply_async(func=afgpy, args=(*argument,)) if isinstance(argument, tuple)
            else pool.apply_async(func=afgpy, args=(argument,)) for argument in range(len(grid_points))]
    pool.close()
    for job in tqdm(jobs):
        par_vals, fluxes = job.get()

        breakpoint()

        h5key = "_".join([str(list(pars.keys())[j])+"_"+str(par_vals[j]) for j in range(len(par_vals))])

        with h5py.File("./fluxes_tophat.h5","a") as f:
            f.create_dataset(h5key, data = fluxes)
        
        parameters.append(par_vals)
        


    finish_time = time.perf_counter()

    with h5py.File("params_tophat.h5", "w") as f:
         f.create_dataset("parameters", data = np.array(parameters))
         f.create_dataset("time", data = afgpy.times)
         f.create_dataset("frequency", data= afgpy.freqs)