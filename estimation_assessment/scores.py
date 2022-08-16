import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os
import sys
import yaml
import h5py

import seaborn as sns
import copy
import re

if __name__=='__main__':
    sys.path.append('./../')
    sys.path.append('./')

from vicon_imu_data_process.dataset import *
from vicon_imu_data_process import process_rawdata as pro_rd

from estimation_models.rnn_models import *

from sklearn.metrics import r2_score, mean_squared_error as mse

from vicon_imu_data_process.const import SAMPLE_FREQUENCY

def calculate_scores(y_true, y_pred):
    '''
    Calculate scores of the estimation value

    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))
    mae = np.mean(abs((y_true - y_pred)))
    r_rmse = rmse / (y_true.max() - y_true.min())

    return round(r2*1000.0)/1000.0, round(rmse*1000.0)/1000.0, round(mae*1000.0)/1000.0, round(r_rmse*1000.0)/1000.0
 
def get_evaluation_metrics(pd_labels, pd_predictions,verbose=0):
    '''
    labels and predictions are pandas dataframe
    return metrics, it is pandas dataframe

    '''

    #i) calculate metrics
    scores={}
    for label in pd_labels.columns:
        scores[label] = list(calculate_scores(pd_labels[label].values, pd_predictions[label].values))
        if(verbose==1):
            print("{}: scores (r2, rmse, mae, r_rmse):".format(label), scores[label][0], scores[label][1], scores[label][2], scores[label][3])

    #ii) shape metrics
    metrics = pd.DataFrame(data=scores, index=['r2','rmse','mae','r_rmse'])
    metrics = metrics.reset_index().rename(columns={'index':'metrics'})
    metrics = metrics.melt(id_vars='metrics',var_name='fields',value_name='scores') 

    return metrics


'''

Calculate the execution time of a model estimation according to the input series 

'''
def calculate_model_time_complexity(model, series, hyperparams):

    window_size = int(hyperparams['window_size'])
    batch_size = int(hyperparams['batch_size'])
    shift_step = int(hyperparams['shift_step'])
    labels_num = int(hyperparams['labels_num'])

    [row_num, column_num] = series.shape

    ds = tf.data.Dataset.from_tensor_slices(series[:, labels_num])

    start = time.time()

    forecast = model.predict(ds)

    end = time.time()

    time_cost = (end - start)/float(row_num)

    return time_cost


'''

 load (best) trained model

'''
def load_trained_model(training_folder, best_model=True):
    trained_model_file = os.path.join(training_folder,'trained_model','my_model.h5')
    #print("Trained model file: ", trained_model_file)
    
    trained_model = tf.keras.models.load_model(trained_model_file)
    
    if(best_model): # load the best model parameter
        best_trained_model_weights = os.path.join(training_folder, "online_checkpoint","cp.ckpt")
        trained_model.load_weights(best_trained_model_weights)
    else:
        print("DO NOT LOAD ITS BEST MODEL!")
        
    return trained_model




'''
Testing model using a trial dataset, specified by input parameters: xy_test
'''
def test_model(training_folder, xy_test, scaler, **kwargs):
    
    #1) create test results folder
    testing_folder = pro_rd.create_testing_files(training_folder)
    
    #2) load hyperparameters, note that the values in hyperparams become string type
    hyperparams_file = os.path.join(training_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr, Loader=yaml.Loader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()

    #3) load trained model
    trained_model = load_trained_model(training_folder)
    
    #4) test data
    model_output, execution_time = model_forecast(trained_model, xy_test, hyperparams)
    model_prediction = model_output.reshape(-1,int(hyperparams['labels_num']))
    
    #5) reshape and inverse normalization
    prediction_xy_test = copy.deepcopy(xy_test) # deep copy of test data
    prediction_xy_test[:,-int(hyperparams['labels_num']):] = model_prediction # using same shape with all datasets
    predictions = scaler.inverse_transform(prediction_xy_test)[:,-int(hyperparams['labels_num']):] # inversed norm predition
    labels  = scaler.inverse_transform(xy_test)[:,-int(hyperparams['labels_num']):]
    features = scaler.inverse_transform(xy_test)[:,:-int(hyperparams['labels_num'])]
    
    #6) save params in testing
    if('test_subject' in kwargs.keys()):
        hyperparams['test_subject_ids_names'] = kwargs['test_subject']
    if('test_trial' in kwargs.keys()):
        hyperparams['test_trial'] = kwargs['test_trial']

    hyperparams['execution_time'] = execution_time

    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    with open(hyperparams_file,'w') as fd:
        yaml.dump(hyperparams, fd)


    #7) transfer testing results' form into pandas Dataframe 
    pd_features = pd.DataFrame(data=features, columns=hyperparams['features_names'])
    pd_labels = pd.DataFrame(data = labels, columns=hyperparams['labels_names'])
    pd_predictions = pd.DataFrame(data = predictions, columns=hyperparams['labels_names'])

    #8) save testing results
    save_test_result(pd_features, pd_labels, pd_predictions, testing_folder)

    return features, labels, predictions, testing_folder, execution_time




'''
save testing results: estimation (predictions) and estiamtion metrics
'''
def save_test_result(pd_features, pd_labels, pd_predictions, testing_folder):
    
    #1) save estiamtion of the testing
    # create testing result (estimation value) file
    saved_test_results_file = os.path.join(testing_folder, "test_results.h5")
    # save tesing results
    with h5py.File(saved_test_results_file,'w') as fd:
        fd.create_dataset('features',data=pd_features.values)
        fd.create_dataset('labels',data=pd_labels.values)
        fd.create_dataset('predictions',data=pd_predictions.values)
        fd['features'].attrs['features_names'] = list(pd_features.columns)
        fd['labels'].attrs['labels_names'] = list(pd_labels.columns)

    #2) save metrics of the estimation results

    # create testing metrics file   
    metrics_file = os.path.join(testing_folder, "test_metrics.csv")

    # calculate metrics
    metrics = get_evaluation_metrics(pd_labels, pd_predictions)
    

    # save metrics
    metrics.to_csv(metrics_file)


'''

Test a trained model on a subject' a trial

'''
def test_model_on_unseen_trial(training_folder, subject_id_name=None,trial='01',hyperparams=None,norm_subjects_trials_data=None, scaler=None):

    #1) load hyperparameters
    if(hyperparams==None):
        hyperparams_file = os.path.join(training_folder, "hyperparams.yaml")
        if os.path.isfile(hyperparams_file):
            fr = open(hyperparams_file, 'r')
            hyperparams = yaml.load(fr,Loader=yaml.Loader)
            fr.close()
        else:
            print("Not Found hyper params file at {}".format(hyperparams_file))
            exit()

    #2) dataset of subject and its trail ------
    if(scaler==None or norm_subjects_trials_data==None):
        subjects_trials_data, norm_subjects_trials_data, scaler = load_normalize_data(hyperparams)

    if subject_id_name == None:
        subject_id_name = hyperparams['test_subject_ids_names'][0]
    if trial==None:
        trial = hyperparams['subjects_trials'][subject_id_name][0]
    
    print("subject: {} and trial: {}".format(subject_id_name, trial))

    try:
        xy_test = norm_subjects_trials_data[subject_id_name][trial]
    except Exception as e:
        print(e, 'Trial: {} is not exist'.format(trial))
    
    # ----------------- estimation
    [features, labels, predictions, testing_folder] = test_model(training_folder, xy_test, scaler, test_subject=subject_id_name,test_trial=trial)

    # ----------- Print  scores of the estimation
    metrics = calculate_scores(labels,predictions)
    print("Estimation metrics: (r2, mae, rmse, r_rmse) ", metrics)
    
    # ----------- Transfer labels and predictions into dadaframe
    pd_labels = pd.DataFrame(labels, columns=hyperparams['labels_names'])
    pd_predictions = pd.DataFrame(predictions, columns=hyperparams['labels_names'])

    return pd_predictions, pd_labels, testing_folder, metrics


'''
Model evaluation:

Use a trained model to estimate labels using an unseen subject' all trials

subject_id_name = None, this means use the test subject specified in hyperprams
trials = None, means use all useful trials in the subject

'''
def test_model_on_unseen_subject(training_folder, subject_id_name=None, trials=None):

    #1) load hyperparameters
    hyperparams_file = os.path.join(training_folder, "hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.Loader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()

    #***TEST SET should not be SYN/ALIGNMENT***
    #hyperparams['syn_features_labels']=False
    #print("WARNING: without synchronization in test")

    #2) load and norm dataset
    subjects_trials_data, norm_subjects_trials_data, scaler = load_normalize_data(hyperparams)

    #3) subject and trials for testing
    #i) the subject for testing
    if(subject_id_name==None):
        subject_id_name = hyperparams['test_subject_ids_names'][0]

    #ii) trials of the subject for testing
    if(trials==None): # use all trials of the subject, the return is lists
        trials = hyperparams['subjects_trials'][subject_id_name]
    if(not isinstance(trials,list)):
        trials = [trials]
    
    testing_results={'labels':[],'predictions': []}
    testing_ingredients = {'subjects': [], 'trials': [],'testing_folder':[]}
    for trial in trials:
        # test of a trial
        [pd_predictions, pd_labels, testing_folder, metrics] = test_model_on_unseen_trial(
                                                training_folder, 
                                                subject_id_name=subject_id_name,
                                                trial=trial,
                                                hyperparams=hyperparams,
                                                norm_subjects_trials_data=norm_subjects_trials_data,
                                                scaler=scaler)
        # add data to list
        testing_ingredients['subjects'].append(subject_id_name)
        testing_ingredients['trials'].append(trial)
        testing_ingredients['testing_folder'].append(testing_folder)

        testing_results['labels'].append(pd_labels)
        testing_results['predictions'].append(pd_predictions)

    return testing_results, testing_ingredients


'''
Model evaluation:

    test multiple models listed in combination_investigation_results on its unseen subject's all trials or specified trials

'''

def evaluate_models_on_unseen_subjects(combination_investigation_results, trials=None):

    # open testing folder
    assessment = []
    line_num = 0
    for line in open(combination_investigation_results,'r'):
        #0) tesing results folder
        if line =='\n':
            continue; # pass space line
        line = line.strip('\n')
        if(line_num==0):
            columns = line.split('\t')
            line_num = line_num + 1
            continue
        try:
            # get testing_folder and test id
            a_single_investigation_config_results = line.split('\t')
            testing_folder = a_single_investigation_config_results[-1]
        except Exception as e:
            pdb.set_trace()

        #1) get folder of the trained model
        #training_folder  = re.search(r".+(\d){2}-(\d){2}",testing_folder).group(0) + "/training" + re.search("_(\d)+", testing_folder).group(0)
        if(re.search('test',os.path.basename(testing_folder))):
            dir_path = os.path.dirname(testing_folder)
            training_folder  = os.path.join(os.path.dirname(dir_path), "training" + re.search("_(\d)+", os.path.basename(dir_path)).group(0))
        else:
            training_folder = testing_folder
            print("Input txt file contains training folders neither test folders")

        print("testing folder:", testing_folder)
        print("training folder:", training_folder)

        #2) testing the model by using all trials of the testing subject (specidied in hyperparams)
        testing_results, testing_ingredients = test_model_on_unseen_subject(training_folder,trials=trials)
        
        #3) estimation results
        try:
            for idx, testing_folder in enumerate(testing_ingredients['testing_folder']):
                #i) collect metric results
                metrics = pd.read_csv(os.path.join(testing_folder,"test_metrics.csv")) # get_evaluation_metrics(pd_labels, pd_predictions)
                metrics['Sensor configurations'] = a_single_investigation_config_results[columns.index('Sensor configurations')]
                metrics['LSTM units'] = a_single_investigation_config_results[columns.index('LSTM units')]
                metrics['Test ID'] =  re.search("test_([0-9])+", testing_folder).group(0)
                metrics['Subjects'] = testing_ingredients['subjects'][idx]
                metrics['Trials'] = testing_ingredients['trials'][idx]
                assessment.append(metrics)
        except Exception as e:
            print(e)
            pdb.set_trace()

    # concate to a pandas dataframe
    pd_assessment = pd.concat(assessment, axis=0)

    # save pandas DataFrame
    overall_metrics_folder = os.path.dirname(combination_investigation_results)
    pd_assessment.to_csv(os.path.join(overall_metrics_folder,"metrics.csv"))
    print('Metrics file save at: {}'.format(overall_metrics_folder))
     
    return pd_assessment


'''
Get estimation and its actual value in a test. The testing results are stored at a folder, testing_folder, which is the only input paramter.
The output are pd_labels, and pd_predictions.

'''
def get_testing_results(testing_folder):

        testing_results = os.path.join(testing_folder, 'test_results.h5')
        try:
            with h5py.File(testing_results,'r') as fd:
                features=fd['features'][:,:]
                predictions=fd['predictions'][:,:]
                labels=fd['labels'][:,:]
                labels_names=fd['labels'].attrs['labels_names']
        except Exception as e:
            print(e)
            print(testing_results)
            pdb.set_trace()

        #2) estimation results
        #i) plot curves
        pd_labels = pd.DataFrame(data = labels, columns = labels_names)
        pd_predictions = pd.DataFrame(data = predictions, columns = labels_names)

        return [pd_labels, pd_predictions] 


'''

Get estimation values of many trials,

Inputs: training_testing_folders:  a txt file (training_result_folders.txt) contains many training or testing configurations and their results.
Return: a pandas dataframe, which containing the actual and estimated values of the tests in training_testing_folder and filtered by selections
'''
    
def get_a_model_test_results(training_testing_folders, selection,**kwargs):
     
    # get test results by re-test trials based this trained model in the training folder: training_***
    if(re.search(r'training_[0-9]{6}', os.path.basename(training_testing_folders))): # training_folder
        training_folder = training_testing_folders
        # testing model in training folder 
        testing_results, testing_ingredients = test_model_on_unseen_subject(training_folder)

     # get test results by searh an exist test folder: testing_result_folders.txt, which contain multiple tests of a trained model
    if(re.search(r'testing_result_folders', os.path.basename(training_testing_folders))): # testing folders
        config_training_testing_folders =  get_investigation_training_testing_folders(training_testing_folders)
        
        # select the necessary testing results using "selection" dictory
        needed_config_training_testing_folders = parase_training_testing_folders(config_training_testing_folders,**selection)
        # get testing results: prediction values in numpy array
        testing_results={'labels': [], 'predictions': []}
        for testing_folder in needed_config_training_testing_folders['testing_folders']:
            #print(testing_folder)
            [pd_labels, pd_predictions] = get_testing_results(testing_folder)
            testing_results['labels'].append(pd_labels)
            testing_results['predictions'].append(pd_predictions)
 
    #i) load actual values
    pd_actual_values = pd.concat(testing_results['labels'], axis=0)
    old_columns = pd_actual_values.columns
    new_columns = ['Actual ' + x for x in old_columns]
    pd_actual_values.rename(columns=dict(zip(old_columns,new_columns)), inplace=True)
 
    #ii) load prediction (estimation) values
    pd_prediction_values = pd.concat(testing_results['predictions'], axis=0)
    old_columns = pd_prediction_values.columns
    new_columns = ['Estimated ' + x for x in old_columns]
    pd_prediction_values.rename(columns=dict(zip(old_columns,new_columns)), inplace=True)
 
    #iii) combine actual and estimation values
    pd_actual_prediction_values = pd.concat([pd_actual_values,pd_prediction_values],axis=1)
    pd_actual_prediction_values.index = pd_actual_prediction_values.index/SAMPLE_FREQUENCY

    #iv) save the values
    pd_actual_prediction_values.to_csv(os.path.dirname(training_testing_folders)+"/actual_prediction_values.csv")

    return pd_actual_prediction_values

'''
Inputs: 
    1. list_training_testing_folders: a list containes several training_result_folders.txt
    2. list_selection: a list of filter parameters to be used to select needed data
Returns:
    a list of dataframe which contains many pd_labels and pd_predictions

'''

def get_multi_models_test_results(list_training_testing_folders, list_selection, **kwargs):
    multi_models_test_results = []
    for training_testing_folders, selection in zip(list_training_testing_folders, list_selection):
        multi_models_test_results.append(get_a_model_test_results(training_testing_folders,selection))

    return multi_models_test_results



'''
Get metrics of list of testing in the integrative investigation.

from the training_testing_folders.

The outputs is assessment dataframe which has r2, mae,rmse, and r_rmse


'''

def get_investigation_assessment(combination_investigation_results):
    
    # open testing folder
    assessment=[]
    line_num=0
    for line in open(combination_investigation_results,'r'):
        #0) tesing results folder
        if line =='\n':
            continue; # pass space line
        line = line.strip('\n')
        if(line_num==0):
            columns = line.split('\t')
            line_num = line_num + 1
            continue

        try:
            # get testing_folder and test id
            a_single_investigation_config_results = line.split('\t')
            testing_folder = a_single_investigation_config_results[-1]
            test_id = re.search("test_([0-9])+", testing_folder).group(0)
        except Exception as e:
            print(e, ', No test folder in the path, Please check {}, and then run evaluate_models_on_unseen_subjects() firstly'.format(combination_investigation_results))
            pdb.set_trace()

        #1) load testing results
        [pd_labels, pd_predictions] = get_testing_results(testing_folder)

        #ii) collect metric results
        try:
            metrics = get_evaluation_metrics(pd_labels, pd_predictions) # get scores
            for idx in range(len(a_single_investigation_config_results)-1): # get investigation configurations
                metrics[columns[idx]] = a_single_investigation_config_results[idx]
            metrics['Test ID'] = test_id # get test id
            # identify the metrics from which trials
            metrics['Metrics ID'] = test_id+'_'+str(re.search("[0-9]+",os.path.basename(testing_folder)).group(0))# get test id


            # read hyperparams 
            hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
            if os.path.isfile(hyperparams_file):
                fr = open(hyperparams_file, 'r')
                hyperparams = yaml.load(fr, Loader=yaml.Loader)
                fr.close()
            else:
                print("Not Found hyper params file at {}".format(hyperparams_file))
                sys.exit()

            if(isinstance(hyperparams['test_subject_ids_names'],list)):
                metrics['Subjects'] = hyperparams['test_subject_ids_names'][0] # few subjects in a list
            else:
                metrics['Subjects'] = hyperparams['test_subject_ids_names'] # only a subject 

            if 'test_trial' in hyperparams.keys():
                if(isinstance(hyperparams['test_trial'],list)):
                    metrics['Trials'] = hyperparams['test_trial'][0]
                else:
                    metrics['Trials'] = hyperparams['test_trial']
            else:
                metrics['Trials'] = 0 # not sure which trials, so set it to 0

            # running time calculation
            if('execution_time' in hyperparams.keys()):
                metrics['execution_time'] = hyperparams['execution_time']
            else:
                metrics['execution_time'] = -100 # Did not define the execution time

            # additional IMUs
            if('additional_imus' in hyperparams.keys()):
                metrics['additional_imus'] = hyperparams['additional_imus']
            else:
                metrics['additional_imus'] = None # Did not define the execution time


        except Exception as e:
            print(e)
            pdb.set_trace()

        assessment.append(metrics)

    #iii) concate to a pandas dataframe
    if(assessment==[]):
        print('results folders are enough')
        exit()
    pd_assessment = pd.concat(assessment, axis=0)

    # data type 
    pd_assessment['syn_features_labels'] = pd_assessment['syn_features_labels'].astype(str).apply(lambda x: True if x=='true' else False)
    pd_assessment['use_frame_index'] = pd_assessment['use_frame_index'].astype(str).apply(lambda x: True if x=='true' else False)

    #3) save pandas DataFrame
    combination_investigation_folder = os.path.dirname(combination_investigation_results)
    pd_assessment.to_csv(os.path.join(combination_investigation_folder, "metrics.csv"))

    return pd_assessment




'''
Get investigation test metrics (r2). 

The metrics are get from all the tests in the integrative investigation


'''

def get_investigation_metrics(combination_investigation_results, metric_fields=['r2']):
    
    #0) calculate assessment metrics
    if(re.search('r2_metrics',os.path.basename(combination_investigation_results))):
        pd_assessment = pd.read_csv(combination_investigation_results, header=0)
        metrics = pd_assessment.loc[pd_assessment['metrics'].isin(metric_fields)]
    elif(re.search('metrics',os.path.basename(combination_investigation_results))):
        pd_assessment = pd.read_csv(combination_investigation_results, header=0)
        metrics = pd_assessment.loc[pd_assessment['metrics'].isin(metric_fields)]
    else:
        pd_assessment = get_investigation_assessment(combination_investigation_results)
        metrics = pd_assessment.loc[pd_assessment['metrics'].isin(metric_fields)]
        
    # reset index
    #r2_metrics.index = np.arange(0,r2_metrics.shape[0])

    #2) add column: IMU number
    metrics.loc[:,'IMU number']=metrics.loc[:,'Sensor configurations'].apply(lambda x: len(x))


    #3) save pandas DataFrame
    combination_investigation_folder = os.path.dirname(combination_investigation_results)
    metrics.to_csv(os.path.join(combination_investigation_folder, "r2_metrics.csv"),index=False)

    return metrics





'''

Get investiagation configurations with training_folders and testing_folders:

    Sensor configurations, IMU number, .... training_folders, testing_folders,
     F, 1, .... ...

THe return, investigation_config_results has two items: 'training_folders' and 'testing_folders'

Each item is pd dataframe contains "traing_testing_folder.txt"  with training_folders or tesing_folders


'''
def get_investigation_training_testing_folders(combination_investigation_results, train_test_id=None):

    # load training and testing results
    investigation_config_results = pd.read_csv(combination_investigation_results,delimiter='\t',header=0)

    # get training and testing folders
    if(re.search('training_(\d){6}', investigation_config_results['training_testing_folders'][0])):
        training_folders = investigation_config_results['training_testing_folders']
        testing_folders = []
        for training_folder in training_folders:
            testing_folders.append(os.path.join(os.path.dirname(training_folder),"testing"+re.search(r'_(\d){6}', training_folder).group(0)))
    elif(re.search('testing_(\d){6}', os.path.basename(investigation_config_results['training_testing_folders'][0]))):
        testing_folders = investigation_config_results['training_testing_folders']
        training_folders = []
        for testing_folder in testing_folders:
            training_folders.append(os.path.join(os.path.dirname(testing_folder),"training"+re.search(r'_(\d){6}', testing_folder).group(0)))
    elif(re.search('test_', os.path.basename(investigation_config_results['training_testing_folders'][0]))):
        testing_folders = investigation_config_results['training_testing_folders']
        training_folders = []
        for testing_folder in testing_folders:
            dir_testing_folder = os.path.dirname(testing_folder)
            training_folders.append(os.path.join(os.path.dirname(dir_testing_folder),"training"+re.search(r'_(\d){6}', dir_testing_folder).group(0)))
    
    # investigation_config_results is a pandas Dataframe
    investigation_config_results['training_folders'] = training_folders
    investigation_config_results['testing_folders'] = testing_folders

    return investigation_config_results



def parase_training_testing_folders(investigation_config_results, landing_manner='all', estimated_variable='all', syn_features_label='both', use_frame_index='both', LSTM_unit='all', IMU_number='all', sensor_configurations='all'):
    
    #2) pick necessary testing or training folders
    if 'landing_manners' in investigation_config_results.columns: # has this investigation
        if landing_manner in set(investigation_config_results['landing_manners']):
            investigation_config_results = investigation_config_results.loc[investigation_config_results['landing_manners']==landing_manner]
        elif(landing_manner=='all'):
            print('ALl landing manners are used')
        else:
            print('specified landing manner is wrong')
            sys.exit()

    if 'estimated_variables' in investigation_config_results.columns: # has this investigation variables
        if estimated_variable in set(investigation_config_results['estimated_variables']): # has this option
            investigation_config_results = investigation_config_results.loc[investigation_config_results['estimated_variables']==estimated_variable]
        elif(estimated_variable=='all'):
            print('ALl estimated variables are used')
        else:
            print('specified estimated variable is not right, it should be: {}'.format(set(investigation_config_results['estimated_variables'])))
            sys.exit()

    if 'syn_features_labels' in investigation_config_results.columns: # has this investigation
        if syn_features_label in set(investigation_config_results['syn_features_labels']):# has this value
            investigation_config_results = investigation_config_results.loc[investigation_config_results['syn_features_labels']==syn_features_label]
            hue=None
        elif(syn_features_label=='both'):
            hue='syn_features_labels'
        else:
            print('syn_features_lable is not right, it should be {}'.format(set(investigation_config_results['syn_features_labels'])))
            sys.exit()

    if 'use_frame_index' in investigation_config_results.columns: # has this investigation variables
        if  use_frame_index in set(investigation_config_results['use_frame_index']): # has right investigation value
            investigation_config_results = investigation_config_results.loc[investigation_config_results['use_frame_index']==use_frame_index]
            hue=None
        elif(use_frame_index=='both'):
            hue='use_frame_index'
        else:
            print('use_frame_index is not right, it should be {}'.format(set(investigation_config_results['use_frame_index'])))
            sys.exit()

    if 'LSTM units' in investigation_config_results.columns: # has this investigation
        if set(LSTM_unit) <= set(investigation_config_results['LSTM units']): # a value of the LSTM unit
            investigation_config_results = investigation_config_results.loc[investigation_config_results['LSTM units'].isin(LSTM_unit)]
        elif(LSTM_unit=='all'):
            print('All LSTM units are used')
        else:
            print('LSTM units is not right, it should be {}'.format(set(investigation_config_results['LSTM units'])))
            sys.exit()

    if 'IMU number' in investigation_config_results.columns: # has this investigation
        if set(IMU_number) <= set(investigation_config_results['IMU number']): # a value of the IMU number
            investigation_config_results = investigation_config_results.loc[investigation_config_results['IMU number'].isin(IMU_number)]
        elif(IMU_number=='all'):
            print('All IMU number are used')
        else:
            print('IMU number is not right, it should be {}'.format(set(investigation_config_results['IMU number'])))
            sys.exit()
    
    if 'Sensor configurations' in investigation_config_results.columns: # has this investigation
        if set(sensor_configurations) <= set(investigation_config_results['Sensor configurations']): # a value of the IMU number
            investigation_config_results = investigation_config_results.loc[investigation_config_results['Sensor configurations'].isin(sensor_configurations)]
        elif(sensor_configurations=='all'):
            print('All sensor configurations are used')
        else:
            print('sensor configurations is not right, it should be {}'.format(set(investigation_config_results['Sensor configurations'])))
            sys.exit()

    return investigation_config_results







if __name__=='__main__':



    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/investigation/2022-05-13/094012/double_KFM_R_syn_5sensor_35testing_result_folders.txt"
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/2022-05-13/001/testing_result_folders.txt"

    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/testing_result_folders.txt"

    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/testing_result_folders.txt"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/10_14/testing_result_folders.txt"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/investigation/2022-05-24/205937/GRFtesting_result_folders.txt"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train_bk/08_30/testing_result_folders.txt"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/new_alignment/testing_result_folders.txt"


    #combination_investigation_metrics = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/investigation/valid_results/metrics.csv"

    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/1_collected_data/study_lstm_units_GRF/3_imu_all_units/testing_result_folders.txt"
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/1_collected_data/study_lstm_units_GRF/3_imu_all_units/testing_result_folders.txt"
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/2_collected_full_cv/2_imu_full_cv_all_trials/testing_result_folders.txt"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/3_imu/3_imu_25_lstm_units/testing_result_folders.txt"
    pd_assessment = evaluate_models_on_unseen_subjects(combination_investigation_results)
    pdb.set_trace()





    ##get_investigation_training_testing_folders(combination_investigation_results)
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/r2_metrics.csv"
    #r2_metrics = get_investigation_metrics(combination_investigation_results)


    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/001/testing_result_folders.txt"
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/001/metrics.csv"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/testing_result_folders.txt"
    r2_metrics = get_investigation_metrics(combination_investigation_results)


    pdb.set_trace()

    #pd_assessment = evaluate_models_on_unseen_subjects(combination_investigation_results)


    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/testing_result_folders.txt"
    investigation_config_results =  get_investigation_training_testing_folders(combination_investigation_results,'training_123113')
    print(investigation_config_results)


    training_folder =  get_investigation_training_testing_folders(combination_investigation_results,'training_123113')
    print(training_folder)
    subject_id_name='P_10_zhangboyuan'
    trial = '01'
    pd_predictions, pd_labels, testing_folder, metrics = test_model_on_unseen_trial(training_folder, 
                                                                                subject_id_name=subject_id_name,
                                                                                trial=trial)
