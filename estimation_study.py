#!/usr/bin/env python
# coding: utf-8
'''
 Import necessary packages

'''
import tensorflow as tf
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

import seaborn as sns
import copy
import re
import json

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH
from vicon_imu_data_process import const


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg

from vicon_imu_data_process.dataset import *
from estimation_models.rnn_models import *


#subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)

cpus=tf.config.list_logical_devices(device_type='CPU')
gpus=tf.config.list_logical_devices(device_type='GPU')
print(cpus,gpus)

'''
Set hyper parameters

'''

def initParameters(labels_names=None,features_names=None):
    # hyper parameters
    hyperparams={}

    # specify labels and features names
    if(labels_names==None):
        labels_names=LABELS_FIELDS
    else:
        labels_names=labels_names

    if(features_names==None):
        features_names=FEATURES_FIELDS
    else:
        features_names=features_names

    # specify other paramters
    columns_names = features_names + labels_names

    hyperparams['features_num'] = len(features_names)
    hyperparams['labels_num'] = len(labels_names)
    hyperparams['features_names'] = features_names
    hyperparams['labels_names'] = labels_names
    hyperparams['columns_names'] = columns_names
    hyperparams['learning_rate'] = 15e-2
    hyperparams['batch_size'] = 40
    hyperparams['window_size'] = 4
    hyperparams['shift_step'] = 1
    hyperparams['epochs'] = 100
    hyperparams['raw_dataset_path'] = os.path.join(DATA_PATH,'features_labels_rawdatasets.hdf5')
    hyperparams['target_leg'] = 'L'
    hyperparams['landing_manner'] = 'double_legs'

    return hyperparams



'''
Main rountine for developing ANN model for biomechanic variable estimations

'''

def train_test_loops(hyperparams=None, fold_number=1, test_multi_trials=False):
    #1) set hyper parameters
    if(hyperparams==None):
        hyperparams = initParameters()
    else:
        hyperparams = hyperparams

    
    #2) create a list of training and testing files
    train_test_loop_folder = os.path.join(RESULTS_PATH, "training_testing", 
                                     str(localtimepkg.strftime("%Y-%m-%d", localtimepkg.localtime())), 
                                     'train_test_loops', # many train and test loop based on cross validation
                                     str(localtimepkg.strftime("%H_%M_%S", localtimepkg.localtime()))
                                    )
    if(os.path.exists(train_test_loop_folder)==False):
        os.makedirs(train_test_loop_folder)   


    # create file for storing train and test folders
    train_test_loop_folders_log = os.path.join(train_test_loop_folder, "train_test_loop_folders.log")
    if(os.path.exists(train_test_loop_folders_log)):
        os.remove(train_test_loop_folders_log)
        
    # create file for storing scores
    cross_val_score_file = os.path.join(train_test_loop_folder, "cross_validation_scores.csv")
    if(os.path.exists(cross_val_score_file)):
        os.remove(cross_val_score_file)
    
    # declare dictory to store training and testing folders
    training_testing_results = {'training_folder':[],'testing_folder':[]}
    cross_val_scores = []
    
    #3) Load and normalize datasets for training and testing
    subjects_trials_data, norm_subjects_trials_data, scaler = load_normalize_data(hyperparams)

    #4) leave-one-out cross-validation
    loo = LeaveOneOut()
    loop_times = 0
    subjects_trials = hyperparams['subjects_trials']
    subject_ids_names = list(subjects_trials.keys())
    cross_scores=[] # scores of cross validation
    # check whether fold number is effective
    if(fold_number>len(subject_ids_names)):
        fold_number = len(subject_ids_names)

    for train_subject_indices, test_subject_indices in loo.split(subject_ids_names):
        #0) select model
        model_type='tf_keras'

        #i) declare model
        model = model_v1(hyperparams)
        #model = model_narx(hyperparams)

        #ii) split dataset
        train_set, valid_set, xy_test = split_dataset(norm_subjects_trials_data, train_subject_indices, test_subject_indices, hyperparams, model_type=model_type, test_multi_trials=test_multi_trials)

        #iii) train model
        trained_model, history_dict, training_folder = train_model(model, hyperparams, train_set, valid_set)
        #trained_model, history_dict, training_folder = train_model_narx(model, hyperparams, train_set, valid_set)
        
        #iv) test model
        if(isinstance(xy_test, list)): # multi trials as test dataset
            for trial_idx, a_trial_xy_test in enumerate(xy_test):
                features, labels, predictions, testing_folder,execution_time = es_as.test_model(training_folder, a_trial_xy_test, scaler)
                training_testing_results['training_folder'].append(training_folder)
                training_testing_results['testing_folder'].append(testing_folder)
                cross_val_scores.append([loop_times, trial_idx] + list(es_as.calculate_scores(labels, predictions)))
        else: # only a trial as test dataset
            features, labels, predictions, testing_folder,execution_time = es_as.test_model(training_folder, xy_test, scaler)
            training_testing_results['training_folder'].append(training_folder)
            training_testing_results['testing_folder'].append(testing_folder)
            cross_val_scores.append([loop_times] + list(es_as.calculate_scores(labels, predictions)))

        loop_times = loop_times + 1
        if loop_times >= fold_number: # only repeat 4 times
           break;# only run a leave-one-out a time

    #5) cross validation scores on test dataset
    cross_scores = np.array(cross_val_scores)
    columns=['fold number', 'test_trial_idx', 'r2','mae','rmse','r_rmse']
    pd_cross_scores = pd.DataFrame(cross_scores, columns=columns[-cross_scores.shape[1]:])
    pd_cross_scores.to_csv(cross_val_score_file)
    print('Cross validation mean r2 scores: {} on fold: {} cross validation on all test trials'.format(pd_cross_scores['r2'].mean(axis=0), fold_number))

    #6) save train and test folder path
    with open(train_test_loop_folders_log, 'w') as fd:
        yaml.dump(training_testing_results, fd)

    #training_testing_results['cross_val_scores'] = cross_scores
    return training_testing_results, xy_test, scaler



if __name__=='__main__':
    pass
    train_test_loops(hyperparams=None, fold_number=1, test_multi_trials=False)
