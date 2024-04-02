import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

import h5py
import os


from sklearn.decomposition import PCA
import json





### import the raw data
fluxes = []

f = h5py.File("../training_data/fluxes_tophat.h5", "r")

for key in f.keys():
    fluxes.append(np.array(f[key]))

f.close()

fluxes = np.array(fluxes)


### rescale it

fluxes = np.log(fluxes)
max_data, min_data = np.max(fluxes,axis =(0,1)), np.min(fluxes, axis = (0,1))


rescaled_data = np.array([((flux-min_data)/(max_data-min_data)).flatten() for flux in fluxes])



### determine the metrics for individual components:

comp_test = {}

for comp in [10, 20, 30, 50, 100]:
    pca = PCA(n_components=comp)
    training_data = pca.fit_transform(rescaled_data)
    print(np.sum(pca.explained_variance_ratio_))

    norm = []
    norm_F = []

    norm_dict = {}

    for j, trainer in enumerate(training_data):
        norm.append(np.abs(pca.inverse_transform(trainer) - rescaled_data[j]).max())
        norm_F.append(np.linalg.norm(pca.inverse_transform(trainer) - rescaled_data[j]))

    norm_dict["max"] = norm
    norm_dict["Frobenius"] = norm_F
    
    comp_test[str(comp)] = norm_dict

    


json_dict = json.dumps(comp_test)

f = open("comp_test.json", "w")
f.write(json_dict)
f.close()