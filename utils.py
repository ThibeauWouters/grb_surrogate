import h5py

### CONSTANTS

filters = ["bessellv", "radio-5.5GHz", "X-ray-5keV", "radio-6GHz",  "radio-3GHz", "X-ray-1keV"]
parameter_names = ['n_ism', 'theta_obs', 'Eiso_c', 'theta_c', 'p_fs', 'eps_e_fs', 'eps_b_fs']

def read_X_file(filename: str):
    with h5py.File(filename, "r") as f:
        X = f["X"][()]
        freqs = f["freqs"][()]
        times = f["times"][()]
    return X, freqs, times

def read_Y_file(filename: str):
    with h5py.File(filename, "r") as f:
        Y = f["Y"][()]
    return Y