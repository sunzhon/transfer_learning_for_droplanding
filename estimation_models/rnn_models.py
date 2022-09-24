#!/usr/bin/env python
'''
 Import necessary packages

'''
import tensorflow as tf
import numpy as np
import pdb
import os
import pandas as pd
import yaml
import h5py
print("tensorflow version:",tf.__version__)
import vicon_imu_data_process.process_landing_data as pro_rd

import copy
import re
import json

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH
from vicon_imu_data_process import const

from vicon_imu_data_process.dataset import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time

# NARX
from sklearn.linear_model import LinearRegression
#from fireTS.models import NARX
import matplotlib.pyplot as plt


'''
Model_V1 definition
'''
def model_v1(hyperparams):
    if(hyperparams['trained_model_folder']==None):
        # new defined model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(int(hyperparams['lstm_units']), 
                                     return_sequences=True, 
                                     activation='tanh',
                                     input_shape=[None,int(hyperparams['features_num'])]
                                    )),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(60,activation='relu'), # linear without activation func
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(30,activation='relu'), # linear without activation func
            tf.keras.layers.Dense(int(hyperparams['labels_num'])) # linear without activation func
        ])
    else: # use trainded model
        trained_model = load_trained_model(hyperparams['trained_model_folder'])
        trained_model.layers[0].trainable=False
        model = trained_model

    return model





'''
Model_V2 definition
'''
def model_v2(hyperparams):
    if(hyperparams['trained_model_folder'] == None): # new defined model
        model = tf.keras.models.Sequential([
        # consider using bidirection LSTM, since we use return_sequences, so the previous state should be also update by considering advanced info.
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(int(hyperparams['lstm_units']), 
                                 return_sequences=True, 
                                 activation='tanh',
                                 input_shape=[None,int(hyperparams['features_num'])]
                                )),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(int(hyperparams['lstm_units']), 
                                 return_sequences=True, 
                                 activation='tanh'
                                )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(60,activation='relu'), # linear without activation func
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30,activation='relu'), # linear without activation func
        tf.keras.layers.Dense(int(hyperparams['labels_num'])) # linear without activation func
    ])
    else:# use trainded model
        trained_model = load_trained_model(hyperparams['trained_model_folder'])
        trained_model.layers[0].trainable=False
        trained_model.layers[1].trainable=False
        model = trained_model

    return model





def my_callbacks(training_folder):
    # online checkpoint callback
    checkpoint_path = training_folder + "/online_checkpoint/cp.ckpt"
    ckp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    #monitor='val_loss',
                                                    verbose=0
                                                    )
    # early stop callback
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-3, mode='min')


    return [ckp_callback, early_callback]





# Model prediction
def model_forecast(model, series, hyperparams):
    window_size = int(hyperparams['window_size'])
    batch_size = int(hyperparams['batch_size'])
    shift_step = int(hyperparams['shift_step'])
    labels_num = int(hyperparams['labels_num'])
    features_num = int(hyperparams['features_num'])


    if(series.shape[1]==(labels_num + features_num)):
        # transfer numpy data into tensors from features
        ds = tf.data.Dataset.from_tensor_slices(series[:,:-labels_num]) # select features from combined data with features and labels
    else:
        ds = tf.data.Dataset.from_tensor_slices(series) # features

    # split datat into windows, window datasets, cannot be used to train model directly
    ds = ds.window(window_size, shift=shift_step, stride=1, drop_remainder = True)
    # split windows into batchs
    ds = ds.flat_map(lambda w: w.batch(window_size)) 
    # batch data set
    ds = ds.batch(1).prefetch(1) # expand dim by axis 0

    '''
    # The model output shape is determined by the shape of the input data
    # The model output has shape (row_num-window_size+1, window_size, labels_num)

    # The model input has shape (batch_num, window_batch_num, window_size, festures_num)
    '''
    st = time.time() # start time
    model_output = model.predict(ds)
    et = time.time() # end time
    dt = round((et-st)/DROPLANDING_PERIOD*1000,1)  # duration time of each frame, unit is ms

    model_prediction=np.row_stack([model_output[0,:,:],model_output[1:,-1,:]])

    # The model prediction shape is (frames of the raw data, labels_num)
    return model_prediction,  dt



