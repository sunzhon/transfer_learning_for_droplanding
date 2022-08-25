#!/usr/bin/env python
# coding: utf-8
'''
 Import necessary packages

'''
import tensorflow as tf
print("tensorflow version:",tf.__version__)
import numpy as np
import pdb
import os
import pandas as pd
import yaml
import h5py

import copy
import re
import json

from vicon_imu_data_process.const import FEATURES_FIELDS, LABELS_FIELDS, DATA_PATH
from vicon_imu_data_process.const import DROPLANDING_PERIOD, RESULTS_PATH
from vicon_imu_data_process import const
from vicon_imu_data_process.dataset import *
import vicon_imu_data_process.process_landing_data as pro_rd
from estimation_models.rnn_models import *
import estimation_assessment.scores as es_as

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg



cpus=tf.config.list_logical_devices(device_type='CPU')
gpus=tf.config.list_logical_devices(device_type='GPU')
print(cpus,gpus)

'''
Set hyper parameters

'''

def initParameters(labels_names=None,features_names=None,data_file='norm_landing_data.hdf5'):
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

    hyperparams['features_names'] = features_names
    hyperparams['labels_names'] = labels_names
    hyperparams['raw_dataset_path'] = os.path.join(DATA_PATH,data_file)

    hyperparams['features_num'] = len(features_names)
    hyperparams['labels_num'] = len(labels_names)
    hyperparams['columns_names'] = columns_names
    hyperparams['learning_rate'] = 8e-2 #15e-2
    hyperparams['batch_size'] = 50 # 40
    hyperparams['window_size'] = 152 # 4
    hyperparams['shift_step'] = 152 # 1
    hyperparams['epochs'] = 10
    hyperparams['subjects_trials'] = pro_rd.get_subjects_trials(hyperparams['raw_dataset_path'])
    hyperparams['target_leg'] = 'R'
    hyperparams['landing_manner'] = 'double_legs'
    hyperparams['lstm_units'] = 40 # 130
    hyperparams['trained_model_folder']=None
    hyperparams['model_selection']='RNN_v1'

    return hyperparams



'''
Main rountine for developing ANN model for biomechanic variable estimations

'''

def train_test_loops(hyperparams=None, test_multi_trials=False):
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
    hyperparams['investigation_results_folder'] = train_test_loop_folder


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
    
    #3) Load normalized dataset and its scaler
    norm_subjects_trials_data = pro_rd.load_subjects_dataset(h5_file_name = hyperparams['raw_dataset_path'],selected_data_fields=hyperparams['columns_names'])
    scaler = pro_rd.load_scaler(hyperparams['scaler_file'])

    #4) leave-one-out cross-validation
    #loo = LeaveOneOut()
    kf = KFold(n_splits=5)
    loop_times = 0
    subjects_trials = hyperparams['subjects_trials']
    subject_ids_names = list(subjects_trials.keys())
    cross_scores=[] # scores of cross validation
    # check whether fold number is effective

    for train_subject_indices, test_subject_indices in kf.split(subject_ids_names):
        #0) select model
        model_type='tf_keras'

        #i) declare model
        if(hyperparams['model_selection']=='RNN_v1'):
            model = model_v1(hyperparams)
        if(hyperparams['model_selection']=='RNN_v2'):
            model = model_v2(hyperparams)
        if(hyperparams['model_selection']=='NARX'):
            model = model_narx(hyperparams)

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
    #5) cross validation scores on test dataset
    cross_scores = np.array(cross_val_scores)
    columns=['fold number', 'test_trial_idx', 'r2','mae','rmse','r_rmse']
    pd_cross_scores = pd.DataFrame(cross_scores, columns=columns[-cross_scores.shape[1]:])
    pd_cross_scores.to_csv(cross_val_score_file)
    print(termcolor.colored('Cross validation mean r2 scores: {} on 5-fold cross validation on all test trials'.format(pd_cross_scores['r2'].mean(axis=0)),'red'))

    #6) save train and test folder path
    with open(train_test_loop_folders_log, 'w') as fd:
        yaml.dump(training_testing_results, fd)

    #training_testing_results['cross_val_scores'] = cross_scores
    print(termcolor.colored("This train and test loop done!",'blue'))
    print(termcolor.colored("Its train and test loop results is in {}".format(train_test_loop_folder),'blue'))
    return training_testing_results, xy_test, scaler



if __name__=='__main__':
    '''
    #1) train model with walking data
    hyperparams = initParameters(labels_names=['R_GRF_Z'],features_names= ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS), data_file = "norm_walking_data.hdf5")
    hyperparams['scaler_file'] = 'walking_scaler_file'
    hyperparams['window_size'] = 152 # 4
    hyperparams['shift_step'] = 152 # 1
    training_test_results, xy_test, scaler = train_test_loops(hyperparams, test_multi_trials=False)
    '''

    #2) fine-tuning model with landing data
    print('Fine tuning model')
    #i) hyperparameter setup
    hyperparams = initParameters(labels_names=['R_GRF_Z'],features_names= ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS),data_file = 'norm_landing_data.hdf5')
    hyperparams['scaler_file'] = 'landing_scaler_file'

    hyperparams['window_size'] = 80 # 4
    hyperparams['shift_step'] = 80 # 1

    #ii) load trained model
    #training_folder = training_test_results['training_folder'][-1]
    #training_folder = "/media/sun/DATA/drop_landing_workspace/results/training_testing/2022-08-19/training_152333"
    #hyperparams['trained_model_folder'] = training_folder

    #iii) fine-tune model
    train_test_loops(hyperparams, test_multi_trials=False)




