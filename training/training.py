import time
import numpy as np
import inspect 
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import os
import sys
sys.path.append("../")
import utils
import h5py
import re

from nmma.em.io import read_photometry_files
from nmma.em.utils import interpolate_nans
from nmma.em.training import SVDTrainingModel
from nmma.em.utils import calc_lc
import nmma.em.model_parameters as model_parameters

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}
MODEL_FUNCTIONS
MODEL_NAME = "afterglowpy_tophat"
data_location = "../preprocessing/"
lcs_dir = "../lcdir/"

####################
### LOADING DATA ###
####################


model_function = MODEL_FUNCTIONS[MODEL_NAME]
X, freqs, times = utils.read_X_file(data_location + "X_afgpy.h5")
times = times / (3600 * 24) # convert to days
Y = utils.read_Y_file(data_location + "Y_afgpy.h5")

filenames = os.listdir(lcs_dir)
full_filenames = [os.path.join(lcs_dir, f) for f in filenames]
print(f"There are {len(full_filenames)} lightcurves for this model.")
data = read_photometry_files(full_filenames)
data = interpolate_nans(data)

keys = list(data.keys())
filts = sorted(list(set(data[keys[0]].keys()) - {"t"}))
print("Filters:")
print(filts)

##################
### SVD object ###
##################

training_data, parameters = model_function(data)
for key in training_data:
    training_data[key]["t"] = times
    
training_model = SVDTrainingModel(MODEL_NAME,
                                  training_data,
                                  parameters,times,
                                  filts,
                                  n_coeff = 10,
                                  interpolation_type="tensorflow",
                                  load_model = False # don't train or load model, just prep the data and exit
)
svd_model = training_model.generate_svd_model()
training_model.svd_model = svd_model

################
### TRAINING ### 
################

n_epochs = 100
for i, filt in enumerate(filts):
    print(f" --- Training for filter {filt}, {(i+1)/len(filts)} ---")
    
    X = training_model.svd_model[filt]['param_array_postprocess']
    n_samples, input_ndim = X.shape
    print(f"Features (input) have shape {X.shape}")

    y = training_model.svd_model[filt]['cAmat'].T
    _, output_ndim = y.shape
    print(f"Labels (output) have shape {y.shape}")

    train_X, val_X, train_y, val_y = train_test_split(X, y, shuffle=True, test_size=0.20, random_state=0)
    
    model = Sequential()
    model.add(
        Dense(
            64,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(train_X.shape[1],),
        )
    )
    model.add(
        Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(train_X.shape[1],),
        )
    )
    model.add(
        Dense(
            64,
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(train_X.shape[1],),
        )
    )
    model.add(Dense(training_model.n_coeff))

    model.compile(optimizer="adam", loss="mse")
    start = time.time()
    training_history = model.fit(
        train_X,
        train_y,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(val_X, val_y),
        verbose=True,
    )
    end = time.time()
    print(f"Time to train: {end - start} seconds = {(end - start) / 60} minutes")
    
    # Save the model as attribute
    training_model.svd_model[filt]["model"] = model
    
    # Plot the loss curves
    train_loss = training_history.history["loss"]
    val_loss = training_history.history["val_loss"]
    plt.figure(figsize=(12, 5))
    plt.plot([i+1 for i in range(len(train_loss))], train_loss, '-o', color="blue", label="Training loss")
    plt.plot([i+1 for i in range(len(val_loss))], val_loss, '-o', color="red", label="Validation loss")
    plt.legend()
    plt.xlabel("Training epoch")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.savefig(f"./figures/training_loss_{filt}.png", bbox_inches="tight")
    plt.close()
 
 
# Finally, save the model       
svd_path = "/home/urash/twouters/new_nmma_models/afterglowpy"
training_model.svd_path = svd_path
training_model.save_model()

print("DONE")