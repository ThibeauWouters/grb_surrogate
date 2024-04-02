import json
import os
import joblib

import numpy as np
import scipy.interpolate as interpolate
#import h5py


class PCATrainingModel(object):
    """
    A training model that will learn to predict spectral fluxes.
    
    """

    def __init__(
        self,
        model,
        data,
        parameters,
        sample_times,
        sample_freqs,
        n_comp,
        pca_path,
    ):
        
        self.model = model
        self.data = data
        self.sample_times = sample_times
        self.n_ts = len(sample_times)
        self.sample_freqs = sample_freqs
        self.n_freqs = len(sample_freqs)
        self.model_parameters = parameters

        self.n_comp = n_comp
        self.pca_path = pca_path

    def get_params_and_data_as_arrays(self):
        logfluxes = []
        params = []
        #from nmma.em.model_parameters import afterglowpy_tophat #TODO how do wanna have the data read in when nmma?
        
        import re
        for j, key in enumerate(self.data.keys()):
            logfluxes.append(np.log(self.data[key]))

            param = [np.abs(float(x)) for x in re.findall(r"[-+]?\d*\.\d+|\d+e[-+]?\d+|\d+", key)]
            param[0] = np.log10(param[0])
            param[2] = np.log10(param[2])
            param[5] = np.log10(param[5])
            param[6] = np.log10(param[6])

            
            params.append(param)

        return np.array(params), np.array(logfluxes)

    def generate_pca_model(self):

        params, logfluxes = self.get_params_and_data_as_arrays()

        #rescale parameters to lie between 0 and 1
        max_params, min_params = np.max(params, axis =0), np.min(params, axis=0)
        self.max_params = max_params
        self.min_params = min_params
        rescaled_params = self.transform_params(params)

        #rescale data to lie between 0 and 1
        max_data, min_data = np.max(logfluxes, axis =(0,1)), np.min(logfluxes, axis = (0,1))
        self.max_data = max_data
        self.min_data = min_data
        rescaled_data = self.transform_data(logfluxes)

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("Scikit-learn is required for PCA.")
            exit()
        
        pca = PCA(n_components=self.n_comp)
        print(f"Fitting PCA model with {self.n_comp} components to the provided data.")
        training_data = pca.fit_transform(rescaled_data)
        print(f"PCA model accounts for a share {np.sum(pca.explained_variance_ratio_)} of the total variance in the data.") #This should be very very close to one.


        self.pca = pca
        self.training_data = training_data
        self.rescaled_params = rescaled_params

        return pca, training_data, rescaled_params






    def transform_params(self, params):

        return (params-self.min_params)/(self.max_params - self.min_params)
    
    def invtransform_params(self, rescaled_params):
        
        return rescaled_params*(self.max_params-self.min_params) + self.min_params
    
    def transform_data(self, data):
 
        return np.array([((entry-self.min_data)/(self.max_data-self.min_data)).flatten() for entry in data])

    def invtransform_data(self, rescaled_data_entry):

        rescaled_data_entry = rescaled_data_entry.reshape(self.n_ts, self.n_freqs)
        logflux = rescaled_data_entry*(self.max_data-self.min_data) + self.min_data
        return logflux
        

         

    def train_tensorflow_model(self, n_epochs, dropout_rate = 0.6):

        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            from sklearn.model_selection import train_test_split
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, Dropout

            import time
        except ImportError:
            print("Please install tensorflow.")
            exit()
        
        print(f" --- Training for spectra ---")

        X = self.rescaled_params
        n_samples, input_ndim = X.shape
        print(f"Features (input) have shape {X.shape}")


        Y = self.training_data
        _, output_ndim = Y.shape
        print(f"Labels (output) have shape {Y.shape}")
        train_X, val_X, train_y, val_y = train_test_split(X, Y, shuffle=True, test_size=0.20, random_state=0)

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
        model.add(Dense(output_ndim))
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

        print(f"Time to train: {end - start} seconds = {(end - start) / 60} minutes.")

        self.tf_model = model
        self.tf_training_history = training_history

    def save_pcamodel(self):
        if not os.path.isdir(self.pca_path):
            os.makedirs(self.pca_path)

       
        #save the tf model
        outfile = os.path.join(self.pca_path, f"{self.model}_tf.h5")
        self.tf_model.save(outfile)

        #save the relevant info for PCA in a dict
        outfile = os.path.join(self.pca_path, f"{self.model}.joblib")
        pca_dict = {}

        pca_dict["n_comp"] = self.n_comp
        pca_dict["mean_"] = self.pca.mean_
        pca_dict["components_"]  = self.pca.components_

        pca_dict["max_data"] = self.max_data
        pca_dict["min_data"] = self.min_data

        pca_dict["max_params"] = self.max_params
        pca_dict["min_params"] = self.min_params
        pca_dict["model_parameters"] = self.model_parameters


        joblib.dump(pca_dict, outfile, compress = 9)

    def load_pcamodel(self):

        #load the tf model
        modelfile = os.path.join(self.pca_path, f"{self.model}_tf.h5")
        try:
            from tensorflow.keras.models import load_model as load_tf_model
        except ImportError:
            print("Please install tensorflow.")
            exit()

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("Scikit-learn is required for PCA.")
            exit()    

        self.tf_model = load_tf_model(modelfile)

        #load the PCA info
        pca_file = os.path.join(self.pca_path, f"{self.model}.joblib")
        pca_dict = joblib.load(pca_file)

        self.pca = PCA(int(pca_dict["n_comp"]))
        self.pca.mean_ = pca_dict["mean_"]
        self.pca.components_ = pca_dict["components_"]

        self.max_data = pca_dict["max_data"]
        self.min_data = pca_dict["min_data"]

        self.max_params = pca_dict["max_params"]
        self.min_params = pca_dict["min_params"]
        self.model_parameters = pca_dict["model_parameters"]



    def calc_lc(self, param, freq):

        rescaled_param = (param-self.min_params)/(self.max_params - self.min_params)
        prediction = self.tf_model.predict(np.array([rescaled_param]))
        
        rescaled_flux = self.pca.inverse_transform(prediction)

        mJys = np.exp(self.invtransform_data(rescaled_flux)).T
        Jys = 1e-3*mJys
        
        Jy = interpolate.interp1d(self.sample_freqs, Jys, axis =1)(freq)

        mag_AB = -48.6 - np.log10(Jy/1e23)*2.5

        return self.sample_times, mag_AB





        



        