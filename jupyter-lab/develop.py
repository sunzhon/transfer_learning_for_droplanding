#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append("./../")




## %load ./../rnn_model.py
#!/usr/bin/env python
'''
 Import necessary packages

'''
import tensorflow as tf
# set hardware config
#tf.debugging.set_log_device_placement(True)

cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

# set gpu memory grouth automatically
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

if(gpus!=[]):
    # set virtal gpu/ logical gpu, create four logical gpu from a physical gpu (gpus[0])
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)
        ]
        )

logical_cpus = tf.config.experimental.list_logical_devices(device_type='CPU')
logical_gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
print('physical cpus and gpus: ',cpus, gpus)
print('physical cpus number: ', len(cpus))
print('physical cpgs number: ', len(gpus))
print('logical cpus and gpus: ',logical_cpus, logical_gpus)
print('logical cpgs number: ', len(logical_gpus))



import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pandas as pd
import yaml
import h5py
print("tensorflow version:",tf.__version__)

import seaborn as sns
import copy
import re
import json
import termcolor


import vicon_imu_data_process.process_landing_data as pro_rd
import estimation_assessment.scores as es_as
import estimation_assessment.visualization as es_vl
from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH, inverse_estimated_variable_dict
from vicon_imu_data_process import const
from vicon_imu_data_process.dataset import *

from estimation_models.rnn_models import *
from estimation_study import *


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg



'''
This function investigate the estimation metrics by 
testing different sensor configurations and model LSTM layer size

'''
def integrative_investigation():

    '''

    # I) baseline
    print(termcolor.colored('Baseline: train a model from scretch with landing data','green'))
    hyperparams = initParameters(labels_names=['R_KNEE_MOMENT_X'],features_names= ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS),data_file = 'kam_norm_landing_data.hdf5')
    hyperparams['scaler_file'] = 'kam_landing_scaler_file'
    hyperparams['model_selection']='RNN_v1'
    hyperparams['trained_model_folder'] = None
    hyperparams['window_size'] = 80 # 4
    hyperparams['shift_step'] = 80 # 1
    training_testing_results, xy_test, scaler = train_test_loops(hyperparams, test_multi_trials=3)
    '''

    # II) transfer learning
    print(termcolor.colored('Train the model with walking data for transfer learning','green'))
    hyperparams = initParameters(labels_names=['R_KNEE_MOMENT_X'],features_names= ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS),data_file = "kam_norm_walking_data.hdf5")
    hyperparams['scaler_file'] = 'kam_walking_scaler_file'
    hyperparams['model_selection']='RNN_v1'
    hyperparams['window_size'] = 80 # 4
    hyperparams['shift_step'] = 80 # 1
    training_testing_results, xy_test, scaler = train_test_loops(hyperparams, test_multi_trials=10)

    #2) fine-tuning model with landing data
    print('Fine tuning a pretrained model')
    #i) hyperparameter setup
    hyperparams = initParameters(labels_names=['R_KNEE_MOMENT_X'],features_names= ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS),data_file = '5subjects_kam_norm_landing_data.hdf5')
    hyperparams['scaler_file'] = '5subjects_kam_landing_scaler_file'
    hyperparams['model_selection']='RNN_v1'
    hyperparams['window_size'] = 80 # 4
    hyperparams['shift_step'] = 80 # 1
    hyperparams['learning_rate'] = 8e-3

    #ii) load trained model
    training_folder = training_testing_results['training_folder'][-1]
    #training_folder = "/media/sun/DATA/drop_landing_workspace/results/training_testing/2022-08-19/training_152333"
    hyperparams['trained_model_folder'] = training_folder
    print(termcolor.colored('Transfer learning by fine-tuning walking model with landing data','green'))
    training_testing_results, xy_test, scaler = train_test_loops(hyperparams, test_multi_trials=10)

    return 0


if __name__ == "__main__":
    combination_investigation_results = integrative_investigation()
