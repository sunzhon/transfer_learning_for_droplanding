import numpy as np
import pandas as pd
import pdb
import os
import sys
import yaml
import h5py

import copy
import re

if __name__=='__main__':
    sys.path.append('./../')
    sys.path.append('./')

#from vicon_imu_data_process.dataset import *
from vicon_imu_data_process import process_landing_data as pro_rd

from sklearn.metrics import r2_score, mean_squared_error as mse
from vicon_imu_data_process.const import SAMPLE_FREQUENCY, RESULTS_PATH

def calculate_scores(y_true, y_pred):
    '''
    Calculate scores of the estimation value

    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))
    mae = np.mean(abs((y_true - y_pred)))
    divider = (y_true.max() - y_true.min())
    if(abs(divider-1e-5)<1e-2):
        r_rmse = 1
    else:
        r_rmse = rmse /divider

    return round(r2*1000.0)/1000.0, round(rmse*1000.0)/1000.0, round(mae*1000.0)/1000.0, round(r_rmse*1000.0)/1000.0
 
def get_evaluation_metrics(pd_labels, pd_predictions, verbose=0):
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

    mean_scores=np.zeros(4)
    for key, data in scores.items(): # calcualte mean scores of multiple labels
        mean_scores += np.array(data)
    mean_scores=mean_scores*1.0/len(scores)
        
    #ii) shape metrics
    #metrics = pd.DataFrame(data=scores, index=['r2','rmse','mae','r_rmse'])
    metrics = pd.DataFrame(data={'label':list(mean_scores)}, index=['r2','rmse','mae','r_rmse']).round(3)
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
def test_model_on_unseen_trial(training_folder, subject_id_name=None,trial='01', hyperparams=None, norm_subjects_trials_data=None, scaler=None):

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
def get_a_test_result(testing_folder):

        try:
            testing_results = os.path.join(testing_folder, 'test_results.h5')
            with h5py.File(testing_results,'r') as fd:
                features=fd['features'][:,:]
                predictions=fd['predictions'][:,:]
                labels=fd['labels'][:,:]
                labels_names=fd['labels'].attrs['labels_names']
        except Exception as e:
            print(e)
            print(testing_results)

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
    
def get_a_model_test_results(training_testing_folders, selection={'unused':None}, **kwargs):
    
    # get training and testing folders into a pandas dataframe
    config_training_testing_folders =  get_investigation_training_testing_folders(training_testing_folders)
    needed_config_training_testing_folders = parase_training_testing_folders(config_training_testing_folders, **selection, **kwargs)

    # get test results by re-test trials based this trained model in the training folder: training_***
    if(re.search(r'training_[0-9]{6}', needed_config_training_testing_folders['training_testing_folders'].iloc[0])!=None): # training_folder
        # testing model in training folder 
        for idx, training_folder in enumerate(needed_config_training_testing_folders['training_testing_folders']):
            testing_results, testing_ingredients = test_model_on_unseen_subject(training_folder)

     # get test results by searh an exist test folder: testing_result_folders.txt, which contain multiple tests of a trained model
    elif(re.search(r'test_[0-9]{6}', needed_config_training_testing_folders['training_testing_folders'].iloc[0])!=None): # testing_folder
        # select the necessary testing results using "selection" dictory
        # get testing results: prediction values in numpy array
        testing_results={'labels': [], 'predictions': [], 'trial_index':[], 'test_subject':[]}
        for idx, testing_folder in enumerate(needed_config_training_testing_folders['testing_folders']):
            [pd_labels, pd_predictions] = get_a_test_result(testing_folder)
            testing_results['labels'].append(pd_labels)
            testing_results['predictions'].append(pd_predictions)
            testing_results['trial_index'].append(pd.DataFrame(data={"trial_index": [idx]*pd_labels.shape[0]}))
            testing_results['test_subject'].append(pd.DataFrame(data={"test_subject": [needed_config_training_testing_folders['test_subject'].iloc[idx]]*pd_labels.shape[0]}))
    else:
        print('training_testing_folders name:{} are wrong, please check that first'.format(training_testing_folders))
        sys.exit()
 
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

    pd_trial_index = pd.concat(testing_results['trial_index'],axis=0)
    pd_test_subject = pd.concat(testing_results['test_subject'],axis=0)
 
    #iii) combine actual and estimation values
    pd_actual_prediction_values = pd.concat([pd_actual_values, pd_prediction_values, pd_trial_index, pd_test_subject],axis=1)
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

def get_list_investigation_results(list_training_testing_folders, list_selection_map=None, **kwargs):
    list_test_results = []
    if list_selection_map!=None:
        for training_testing_folders, selection in zip(list_training_testing_folders, list_selection_map):
            list_test_results.append(get_a_model_test_results(training_testing_folders, selection, **kwargs))
    else:
        for training_testing_folders in list_training_testing_folders:
            list_test_results.append(get_a_model_test_results(training_testing_folders, **kwargs))

    # transform to dataframe
    for idx in range(len(list_test_results)):
        list_test_results[idx]['test_folder_index'] = idx
    
    return pd.concat(list_test_results,axis=0)



'''
Get metrics of list of testing in the integrative investigation.

from the training_testing_folders.

The outputs is assessment dataframe which has r2, mae,rmse, and r_rmse


'''

def survey_investigation_assessment(combination_investigation_results):
    
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
        pdb.set_trace()
        try:
            # if there is no r2 in "training_testing_folders.txt file"
            if('r2' not in columns):
                [pd_labels, pd_predictions] = get_a_test_result(testing_folder)
                metrics = get_evaluation_metrics(pd_labels, pd_predictions) # get scores
            else:
                metrics={}

            #ii) collect metric results
            for idx in range(len(a_single_investigation_config_results)-1): # get investigation configurations
                metrics[columns[idx]] = a_single_investigation_config_results[idx]

            metrics['Test ID'] = test_id # get test id
            # identify the metrics from which trials
            metrics['Metrics ID'] = test_id+'_'+str(re.search("[0-9]+",os.path.basename(testing_folder)).group(0))# get test id
            # get imu sensors 
            features_name = metrics['features_name']
            pattern = re.compile(r"L_FOOT|R_FOOT|L_SHANK|R_SHANK|L_THIGH|R_THIGH|WAIST|CHEST")
            imus = " ".join(list(set(pattern.findall("".join(features_name)))))
            imus = re.sub("R_SHANK", 'rS', imus)
            imus = re.sub("L_SHANK", 'lS', imus)
            imus = re.sub("R_THIGH", 'rT', imus)
            imus = re.sub("L_THIGH", 'lT', imus)
            imus = re.sub("R_FOOT", 'rF', imus)
            imus = re.sub("L_FOOT", 'lF', imus)
            imus = re.sub("CHEST", 'C', imus)
            imus = re.sub("WAIST", 'W', imus)
            metrics['IMUs'] = imus

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
                metrics['subjects'] = hyperparams['test_subject_ids_names'][0] # few subjects in a list
            else:
                metrics['subjects'] = hyperparams['test_subject_ids_names'] # only a subject 

            if 'test_subject' in hyperparams.keys():
                metrics['test_subject'] = hyperparams['test_subject']

            if 'test_trial' in hyperparams.keys():
                if(isinstance(hyperparams['test_trial'],list)):
                    metrics['trials'] = hyperparams['test_trial'][0]
                else:
                    metrics['trials'] = hyperparams['test_trial']
            else:
                metrics['trials'] = 0 # not sure which trials, so set it to 0

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
        
    if(isinstance(assessment[0],dict)):
        pd_assessment = pd.DataFrame(assessment)
    else:
        pd_assessment = pd.concat(assessment, axis=0)

    # data type 
    if('syn_features_labels' in list(pd_assessment.columns)):
        pd_assessment['syn_features_labels'] = pd_assessment['syn_features_labels'].astype(str).apply(lambda x: True if x=='true' else False)
    if('use_frame_index' in list(pd_assessment.columns)):
        pd_assessment['use_frame_index'] = pd_assessment['use_frame_index'].astype(str).apply(lambda x: True if x=='true' else False)
    if('r2' in list(pd_assessment.columns)):
        pd_assessment['r2'] = pd_assessment['r2'].astype(float)
    if('r_rmse' in list(pd_assessment.columns)):
        pd_assessment['r_rmse'] = pd_assessment['r_rmse'].astype(float)
    if('rmse' in list(pd_assessment.columns)):
        pd_assessment['rmse'] = pd_assessment['rmse'].astype(float)
    if('mae' in list(pd_assessment.columns)):
        pd_assessment['mae'] = pd_assessment['mae'].astype(float)

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
    # 0-i) read metrics file if exists
    if(re.search('r2_metrics', os.path.basename(combination_investigation_results))):
        pd_assessment = pd.read_csv(combination_investigation_results, header=0)
    elif(re.search('metrics', os.path.basename(combination_investigation_results))):
        pd_assessment = pd.read_csv(combination_investigation_results, header=0)
    else: #0-iii) survey metrics from testing folders
        pd_assessment = survey_investigation_assessment(combination_investigation_results)

    #1) get metrics
    if('metrics' in pd_assessment.columns):
        metrics = pd_assessment.loc[pd_assessment['metrics'].isin(metric_fields)]
    else:
        metrics = pd_assessment

        
    # reset index
    #r2_metrics.index = np.arange(0,r2_metrics.shape[0])

    #2) add column: IMU number
    if('Sensor configurations' in metrics.columns):
        metrics.loc[:,'IMU number']=metrics.loc[:,'Sensor configurations'].apply(lambda x: len(x))


    #3) save pandas DataFrame
    combination_investigation_folder = os.path.dirname(combination_investigation_results)
    metrics.to_csv(os.path.join(combination_investigation_folder, "r2_metrics.csv"),index=False)

    return metrics


"""

Get list of combination_investigation_results

"""
def get_list_investigation_metrics(list_combination_investigation_results, metric_fields=['r2']):
    if(not isinstance(list_combination_investigation_results,list)):
        list_combination_investigation_results = [list_combination_investigation_results]

    list_metrics=[]
    for combination_investigation_result in list_combination_investigation_results:
        print(combination_investigation_result)
        list_metrics.append(get_investigation_metrics(combination_investigation_result,metric_fields))
        metrics=pd.concat(list_metrics,axis=0)

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


"""

Parase list of combination_investigation_results

"""
def parse_list_investigation_metrics(list_combination_investigation_results, calculate_mean_subject_r2=False, **filters):

    if(not isinstance(list_combination_investigation_results,list)):
        list_combination_investigation_results = [list_combination_investigation_results]

    list_metrics=[]
    for combination_investigation_result in list_combination_investigation_results:
        print(combination_investigation_result)
        list_metrics.append(parse_metrics(combination_investigation_result, 
                                             **filters
                                            ))
        metrics=pd.concat(list_metrics,axis=0)

    if(calculate_mean_subject_r2):
        # calculate mean subject r2
        metrics['mean_subject_r2']=0
        mean_subject_r2 = metrics[['subject_num', 'train_sub_num', 'trial_num','test_subject','r2','model_selection']].groupby(['subject_num', 'train_sub_num', 'trial_num','test_subject','model_selection']).mean().round(4)
        # add mean subject metrics into metrics
        for subject_num in set(metrics['subject_num']):
            for train_sub_num in set(metrics['train_sub_num']):
                for trial_num in set(metrics['trial_num']): 
                    for test_subject in set(metrics['test_subject']):
                        for model_selection in set(metrics['model_selection']):
                            if(test_subject in mean_subject_r2.loc[subject_num, trial_num].index):
                                metrics.loc[(metrics['subject_num']==subject_num) & (metrics['train_sub_num']==train_sub_num) & (metrics['trial_num']==trial_num) & (metrics['test_subject']==test_subject) & (metrics['model_selection']==model_selection), 'mean_subject_r2']=mean_subject_r2.loc[subject_num, trial_num, test_subject, model_selection].values[0]

    return metrics


'''
 Explain the plot configuration to generate necessary data

'''


def parse_metrics(combination_investigation_results, landing_manner='all', estimated_variable='all', syn_features_label='both', use_frame_index='both', LSTM_unit='all', sensor_config='all',IMU_number='all', drop_value=None, metric_fields=['r2'],sort_variable=None, **kwargs):

    #1) load assessment metrics
    metrics = get_investigation_metrics(combination_investigation_results,metric_fields=metric_fields)
    if('scores'in metrics.columns):
        metrics['scores'] = metrics['scores'].astype(float)
    elif('r2' in metrics.columns):
        metrics['r2'] =  metrics['r2'].astype(float)
        
    # drop some cases (test)
    metrics.index = np.arange(0, metrics.shape[0])
    if(drop_value!=None):
        if(('metrics' in metrics.columns) and ('scores' in metrics.columns)):
            selected_data = metrics.loc[(metrics['metrics']=='r2') & (metrics['scores']>=drop_value)]
            if('Metrics ID' in metrics.columns): # if metrics has column: 'Metrics ID'
                metrics = metrics.loc[metrics['Metrics ID'].isin(selected_data['Metrics ID'])]
            else:
                metrics = metrics.drop(metrics[metrics['scores']<drop_value].index)
            print('DROP R2 cases below :{}'.format(drop_value))
        elif('r2' in metrics.columns):
            metrics = metrics.drop(metrics[metrics['r2']<drop_value].index)
            print('DROP R2 cases below :{}'.format(drop_value))

    #2) pick necessary metrics
    if 'landing_manners' in metrics.columns: # has this investigation
        if landing_manner in set(metrics['landing_manners']):
            metrics = metrics.loc[metrics['landing_manners']==landing_manner]
        elif(landing_manner=='all'):
            print('ALl landing manners are used')
        else:
            print('specified landing manner is wrong for the {}'.format(combination_investigation_results))

    if 'estimated_variables' in metrics.columns: # has this investigation variables
        if estimated_variable in set(metrics['estimated_variables']): # has this option
            metrics = metrics.loc[metrics['estimated_variables']==estimated_variable]
        elif(estimated_variable=='all'):
            print('ALl estimated variables are used')
        else:
            print('specified estimated variable is not right, it should be: {}'.format(set(metrics['estimated_variables'])))

    if 'syn_features_labels' in metrics.columns: # has this investigation
        if syn_features_label in set(metrics['syn_features_labels']):# has this value
            metrics = metrics.loc[metrics['syn_features_labels']==syn_features_label]
            hue=None
        elif(syn_features_label=='both'):
            hue='syn_features_labels'
        else:
            print('syn_features_lable is not right, it should be {}'.format(set(metrics['syn_features_labels'])))

    if 'use_frame_index' in metrics.columns: # has this investigation variables
        if  use_frame_index in set(metrics['use_frame_index']): # has right investigation value
            metrics = metrics.loc[metrics['use_frame_index']==use_frame_index]
            hue=None
        elif(use_frame_index=='both'):
            hue='use_frame_index'
        else:
            print('use_frame_index is not right, it should be {}'.format(set(metrics['use_frame_index'])))
            sys.exit()

    if 'LSTM units' in metrics.columns: # has this investigation
        if set(LSTM_unit) <= set(metrics['LSTM units']): # a value of the LSTM unit
            metrics = metrics.loc[metrics['LSTM units'].isin(LSTM_unit)]
        elif(LSTM_unit=='all'):
            print('All LSTM units are used')
        else:
            print('LSTM units is not right, it should be {}'.format(set(metrics['LSTM units'])))
            #sys.exit()

    if 'Sensor configurations' in metrics.columns: # has this investigation
        if set(sensor_config) <= set(metrics['Sensor configurations']): # a value of the IMU number
            metrics = metrics.loc[metrics['Sensor configurations'].isin(sensor_config)]
        elif(sensor_config=='all'):
            print('All sensor configurations are used')
        else:
            print('sensor configurations is not right, it should be {}'.format(set(metrics['Sensor configurations'])))
            #sys.exit()

    if 'IMU number' in metrics.columns: # has this investigation
        if set(IMU_number) <= set(metrics['IMU number']): # a value of the IMU number
            metrics = metrics.loc[metrics['IMU number'].isin(IMU_number)]
        elif(IMU_number=='all'):
            print('All IMU number are used')
        else:
            print('IMU number is not right, it should be {}'.format(set(metrics['IMU number'])))
            #sys.exit()
        metrics['IMU number'] = metrics['IMU number'].astype(str)

    #3) add average scores of each sensor configurations
    if 'Sensor configurations' in metrics.columns: # has this investigation
        metrics['average scores'] = 0.0
        mean_scores_of_sensors = metrics.groupby('Sensor configurations').median()
        for sensor_config in list(set(metrics['Sensor configurations'])):
            metrics.loc[metrics['Sensor configurations']==sensor_config,'average scores'] = mean_scores_of_sensors.loc[sensor_config, 'scores']

    #5) sort value
    if(sort_variable!=None):
        metrics[sort_variable] = metrics[sort_variable].astype('float64')
        metrics.sort_values(by=[sort_variable], ascending=True, inplace=True)
        #metrics[sort_variable] = metrics[sort_variable].astype('int')


    #6) other fliters
    if kwargs!=None:
        for key, value in kwargs.items():
            if(key in metrics.columns):
                if set(value) <= set(metrics[key]): # a value of the test id
                    metrics = metrics.loc[metrics[key].isin(value)]
                elif(value=='all'):
                    print('All {} are used'.format(key))
                else:
                    print('{} is not right, it should be {}'.format(key, set(metrics[key])))
                    sys.exit()
    

    return metrics


def parase_training_testing_folders(investigation_config_results, landing_manner='all', estimated_variable='all', syn_features_label='both', use_frame_index='both', LSTM_unit='all', IMU_number='all', sensor_configurations='all', test_id='all', **kwargs):
    
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
            print('specified estimated variable is not right, it should be a list included one or some of these: {}'.format(set(investigation_config_results['estimated_variables'])))
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
            print('sensor configurations is not right, it should be a list included one of some of these {}'.format(set(investigation_config_results['Sensor configurations'])))
            sys.exit()
    if 'Test ID' in investigation_config_results.columns: # has this investigation
        if set(test_id) <= set(investigation_config_results['Test ID']): # a value of the test id
            investigation_config_results = investigation_config_results.loc[investigation_config_results['Test ID'].isin(test_id)]
        elif(test_id=='all'):
            print('All test_id are used')
        else:
            print('test_id is not right, it should be a list include one or some of these {}'.format(set(investigation_config_results['Test ID'])))
            sys.exit()

    # other fliters
    if kwargs!=None:
        for key, value in kwargs.items():
            if(key in investigation_config_results.columns):
                if set(value) <= set(investigation_config_results[key]): # a value of the test id
                    investigation_config_results = investigation_config_results.loc[investigation_config_results[key].isin(value)]
                elif(value=='all'):
                    print('All {} are used'.format(key))
                else:
                    print('{} and {} is not right, it should be a list include one of some of these {}'.format(key, value, set(investigation_config_results[key])))
                    sys.exit()

    return investigation_config_results







if __name__=='__main__':

    combination_investigation_results = [
            os.path.join(RESULTS_PATH, "training_testing", "baseline_mlnn_t2","testing_result_folders.txt"),
            #os.path.join(RESULTS_PATH, "training_testing", "baseline_mlnn_v12","testing_result_folders.txt"),
            #os.path.join(RESULTS_PATH, "training_testing", "augmentation_v12","testing_result_folders.txt"),
            ]


    #1) get testing results: estimation and ground truth
    list_selections = [{'train_index': 0, 
        'test_subject':['P_10_dongxuan']
        }
    ]


    test_results = get_list_investigation_results(combination_investigation_results, list_selections)
    pdb.set_trace()

    exit()


    # plot curves
    combination_investigation_results = ["/media/sun/DATA/drop_landing_workspace/results/training_testing/baseline/testing_result_folders.txt",
                                 "/media/sun/DATA/drop_landing_workspace/results/training_testing/dann/testing_result_folders.txt"
                                ]
    selection={'child_test_id': ['test_1']}
    #test_results = get_list_investigation_results([combination_investigation_results[0]], **selection)

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
    #pd_assessment = evaluate_models_on_unseen_subjects(combination_investigation_results)




    ##get_investigation_training_testing_folders(combination_investigation_results)
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/r2_metrics.csv"
    #r2_metrics = get_investigation_metrics(combination_investigation_results)


    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/001/testing_result_folders.txt"
    #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/latest_train/001/metrics.csv"
    combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/testing_result_folders.txt"
    combination_investigation_results =  [os.path.join(RESULTS_PATH, "training_testing", "augmentation_dkem_v5",str(rot_id)+'rotid', str(sub_num)+"sub", str(trial_num)+"tri", "testing_result_folders.txt") for sub_num in range(14,15,1) for trial_num in range(25, 26,5) for rot_id in [6]]
    print(combination_investigation_results)
    r2_metrics = get_list_investigation_metrics(combination_investigation_results)

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
