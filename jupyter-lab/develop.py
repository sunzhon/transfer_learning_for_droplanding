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
import vicon_imu_data_process.process_rawdata as pro_rd
import estimation_assessment.scores as es_as
import estimation_assessment.visualization as es_vl

import seaborn as sns
import copy
import re
import json

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH, inverse_estimated_variable_dict
from vicon_imu_data_process import const
from vicon_imu_data_process.dataset import *
from vicon_imu_data_process.process_rawdata import *


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

def integrative_investigation(investigation_variables, prefix_name='', test_multi_trials=False,**kwargs):

    #1) paraser investigation variables
    sensor_configurations = investigation_variables['sensor_configurations']
    lstm_units  = investigation_variables['lstm_units']
    window_sizes = investigation_variables['window_size']
    shift_step = investigation_variables['shift_step']
    additional_IMUs = investigation_variables['additional_IMUs']

    if('syn_features_labels' in investigation_variables.keys()):
        syn_features_labels = investigation_variables['syn_features_labels']
    else:
        syn_features_labels = [False] # default value is false, to synchorinize features and labels using event


    if('use_frame_index' in investigation_variables.keys()):
        use_frame_index = investigation_variables['use_frame_index']
    else:
        use_frame_index = [True] # default value is true, to induce frame index to input the model

    estimated_variables = investigation_variables['estimated_variables']
    landing_manners = investigation_variables['landing_manners']
    target_leg = investigation_variables['target_leg']

    # init hyper parameters
    hyperparams = initParameters()

    # create forlder to store this investigation
    investigation_results_folder = os.path.join(RESULTS_PATH, "training_testing", 
                                     str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())), 
                                     str(localtimepkg.strftime("%H_%M", localtimepkg.localtime())),
                                    )
    # save investigation folders
    if(os.path.exists(investigation_results_folder)==False):
        os.makedirs(investigation_results_folder)   
    hyperparams['investigation_results_folder'] = investigation_results_folder
    
    #2) train and test model
    combination_investigation_info = []

    # default parameters
    if 'fold_number' in kwargs.keys():
        fold_number = kwargs['fold_number']
    else:
        fold_number = 1 # default

    #i) drop landing manners, landing manners: single-leg or double-leg drop landing
    for landing_manner in landing_manners:
        hyperparams['landing_manner'] = landing_manner # single or double legs
        hyperparams['target_leg'] = target_leg
        if landing_manner not in ['double_legs','single_leg']:
            print('landing manner is wrong')
            exit()

        #ii) synchronization state
        for syn_state in syn_features_labels:
            hyperparams['syn_features_labels'] = syn_state

            #iii) use frame index as input for model
            for use_frame_index_state in use_frame_index:
                hyperparams['use_frame_index'] = use_frame_index_state

                #iii) init hyper params with different labels
                for estimated_variable in estimated_variables:
                    labels_fields = [hyperparams['target_leg'] + '_' + x for x in estimated_variable]
                    hyperparams['labels_names'] = labels_fields
                    hyperparams['labels_num'] = len(hyperparams['labels_names'])
                    hyperparams['abbrev_estimated_variables'] = [inverse_estimated_variable_dict[x] for x in estimated_variable]

                    #iv) sensor configurations
                    for sensor_configuration_name, sensor_list in sensor_configurations.items():
                        hyperparams['sensor_configurations'] = sensor_configuration_name
                        # features fields based on sensors
                        features_fields = const.extract_imu_fields(sensor_list, const.ACC_GYRO_FIELDS)

                        if(use_frame_index_state): # time and imu field with direction of right and left leg
                            hyperparams['features_names'] = ['TIME'] + [hyperparams['target_leg'] + '_' + x if(re.search('(FOOT)|(SHANK)|(THIGH)',x)) else x for x in features_fields]
                        else:
                            hyperparams['features_names'] = [hyperparams['target_leg']+'_'+x if(re.search('(FOOT)|(SHANK)|(THIGH)',x)) else x for x in features_fields]

                        # temp feature names
                        temp_feature_names = copy.deepcopy(hyperparams['features_names'])
                        for additional_IMU_config, add_imus in additional_IMUs.items():
                            hyperparams['additional_imus'] = additional_IMU_config
                            if(add_imus!=[]): # additional IMUs
                                hyperparams['features_names'] = temp_feature_names + const.extract_imu_fields(add_imus, const.ACC_GYRO_FIELDS)
                            
                            # feature names and columns names
                            hyperparams['features_num'] = len(hyperparams['features_names'])
                            hyperparams['columns_names'] = hyperparams['features_names'] + hyperparams['labels_names']

                            # set subjects and trials
                            hyperparams['raw_dataset_path'] = os.path.join(DATA_PATH,'walking_data.hdf5')
                            hyperparams['subjects_trials'] = pro_rd.get_subjects_trials(hyperparams['raw_dataset_path'])

                            #v) model size configuations
                            for lstm_unit in lstm_units:
                                hyperparams['lstm_units'] = lstm_unit

                                for window_size in window_sizes:
                                    hyperparams['window_size'] = window_size
                                    hyperparams['shift_step'] = shift_step

                                    # configuration list name
                                    a_single_config_columns = '\t'.join(['Sensor configurations',
                                                                         'LSTM units',
                                                                         'syn_features_labels',
                                                                         'use_frame_index',
                                                                         'landing_manners',
                                                                         'window_size',
                                                                         'estimated_variables',
                                                                         'additional_IMUs',
                                                                         'training_testing_folders',
                                                                        ]) + '\n'

                                    # train and test model
                                    print("#**************************************************************************#")
                                    print("Investigation configs: {}".format(a_single_config_columns))
                                    print("Investigation configs: {}".format([sensor_configuration_name, 
                                                                              str(lstm_unit), 
                                                                              str(syn_state), 
                                                                              str(use_frame_index_state), 
                                                                              landing_manner, 
                                                                              window_size,
                                                                              additional_IMU_config,
                                                                              estimated_variable
                                                                             ]))

                                    # ******DO TRAINING AND TESTING****** #
                                    training_testing_folders, xy_test, scaler =  train_test_loops(
                                        hyperparams, 
                                        fold_number=fold_number, 
                                        test_multi_trials=test_multi_trials
                                    )# model traning

                                    # list testing folders 
                                    a_single_config = [sensor_configuration_name, 
                                                       str(lstm_unit), 
                                                       str(syn_state), 
                                                       str(use_frame_index_state), 
                                                       landing_manner, 
                                                       window_size,
                                                       hyperparams['abbrev_estimated_variables'], 
                                                       additional_IMU_config,
                                                       training_testing_folders
                                                      ]
                                    combination_investigation_info.append(a_single_config)

    #3) create folders to save testing folders
    combination_investigation_testing_folders = os.path.join(RESULTS_PATH,"investigation",
                                         str(localtimepkg.strftime("%Y-%m-%d",localtimepkg.localtime())),
                                         str(localtimepkg.strftime("%H%M%S", localtimepkg.localtime())))
    if(not os.path.exists(combination_investigation_testing_folders)):
        os.makedirs(combination_investigation_testing_folders)
    
    #4) save testing folders
    combination_testing_folders = os.path.join(combination_investigation_testing_folders, prefix_name + "testing_result_folders.txt")
    if(os.path.exists(combination_testing_folders)==False):
        with open(combination_testing_folders,'a') as f:
            f.write(a_single_config_columns)
            for single_investigation_info in combination_investigation_info:
                train_test_results = single_investigation_info[-1] # the last elements is a list
                for idx, testing_folder in enumerate(train_test_results["testing_folder"]): # in a loops which has many train and test loop
                    # a single investigation config info and its results
                    single_investigation_info_results = single_investigation_info[:-1] + [testing_folder]
                    # transfer into strings with '\t' seperator
                    str_single_investigation_info_results = '\t'.join([str(i) for i in single_investigation_info_results])
                    # save config and its results
                    f.write(str_single_investigation_info_results +'\n')
                                                    
    print("INESTIGATION DONE!")
    return combination_testing_folders


# ## Perform investigation by training model


#1) The variables that are needed to be investigate
investigation_variables={
    "sensor_configurations":
                            {
                             #    'F': ['FOOT'],
                             #    'S': ['SHANK'],
                             #    'T': ['THIGH'],
                             #    'W': ['WAIST'],
                             #    'C': ['CHEST']

                             #   'FS': ['FOOT','SHANK'],
                             #   'FT': ['FOOT','THIGH'],
                             #   'FW': ['FOOT','WAIST'],
                             #   'FC': ['FOOT','CHEST'],
                             #   'ST': ['SHANK','THIGH'],
                             #   'SW': ['SHANK','WAIST'],
                             #   'SC': ['SHANK','CHEST'],
                             #   'TW': ['THIGH','WAIST'], 
                             #   'TC': ['THIGH', 'CHEST'],
                             #   'WC': ['WAIST', 'CHEST']


                             #   'FST': ['FOOT','SHANK','THIGH'], 
                             #   'FSW': ['FOOT','SHANK','WAIST'],
                             #   'FSC': ['FOOT','SHANK','CHEST'],
                             #   'FTW': ['FOOT','THIGH','WAIST'],
                             #   'FTC': ['FOOT','THIGH','CHEST'],
                             #   'FWC': ['FOOT','WAIST', 'CHEST'],
                             #   'STW': ['SHANK','THIGH','WAIST' ],
                             #   'STC': ['SHANK','THIGH','CHEST' ],
                             #   'SWC': ['SHANK','WAIST','CHEST' ],
                             #   'TWC': ['THIGH','WAIST', 'CHEST']

                             #   'FSTW': ['FOOT','SHANK','THIGH','WAIST'], 
                             #   'FSTC': ['FOOT','SHANK','THIGH','CHEST'] # optimal config for vGRF during double-leg drop landing
                             #   'FSWC': ['FOOT','SHANK','WAIST', 'CHEST'],
                             #   'FTWC': ['FOOT','THIGH','WAIST', 'CHEST'],
                             #   'STWC': ['SHANK','THIGH','WAIST', 'CHEST']

                               'FSTWC': ['FOOT','SHANK','THIGH','WAIST', 'CHEST']
                             #   'None':[]
                             },
    
    #"syn_features_labels": [True, False],
    "syn_features_labels": [False],
    #"use_frame_index": [True, False],
    "use_frame_index": [True],
    #'estimated_variables': [['KNEE_MOMENT_X'], ['GRF_Z']],  # KFM, KAM, GRF
    'estimated_variables': [['KNEE_MOMENT_X']],
    #"landing_manners": [ 'double_legs', 'single_leg'],
    "landing_manners": ['double_legs'],
    "window_size" :[4],
    "shift_step" : 1,
    
    "lstm_units": [130],
    'target_leg': 'R',
    'additional_IMUs': {
     #   'F': ['L_FOOT'],
     #   'S': ['L_SHANK'],
     #   'T': ['L_THIGH'],
     #   'FS': ['L_FOOT','L_SHANK'],
     #   'ST': ['L_SHANK','L_THIGH'],
     #   'FT': ['L_FOOT','L_THIGH'],
        'FST':['L_FOOT','L_SHANK','L_THIGH']
     #   'None':[]
    }

}


if __name__ == "__main__":
    #2) investigate model
    combination_investigation_results = integrative_investigation(investigation_variables,prefix_name='GRF_X',fold_number=2, test_multi_trials=True)

    print(combination_investigation_results)
