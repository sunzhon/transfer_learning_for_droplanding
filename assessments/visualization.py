#! /bin/pyenv python
#coding: --utf-8

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
from statannotations.Annotator import Annotator



current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../../../")
sys.path.append(added_path)
from CRCF.plot_utilities import *

#sys.path.append(os.getenv('STPY_WORKSPACE'))
#if os.getenv("STPY_WORKSPACE")!=None:
#    from CRCF.plot_utilities import *


current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../")
sys.path.append(added_path)
from vicon_imu_data_process.const import SAMPLE_FREQUENCY
from vicon_imu_data_process.const import RESULTS_PATH



if __name__ == '__main__':
    from scores import *
else:
    from assessments.scores import *

'''
Plot the estimation results

'''
def plot_prediction(pd_labels, pd_predictions, testing_folder,**kwargs):

    #i) load hyper parameters
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()

    #ii) create file name to save plot results
    test_subject_ids_names = hyperparams['test_subject_ids_names']
    prediction_file = os.path.join(testing_folder, test_subject_ids_names[0] + '_estimation.svg')

    #iii) plot the estimation results and errors
    plot_actual_estimation_curves(pd_labels, 
                                    pd_predictions, 
                                    testing_folder,
                                    figtitle=prediction_file,
                                    **kwargs)




def plot_prediction_statistic(features, labels, predictions,testing_folder):
    '''
    This function calculate the error between predicted and ground truth, and plot them for comparison
    '''
    
    # load hyperparameters, Note the values in hyperparams become string type
    hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    
    # hyper parameters    
    features_names=hyperparams['features_names']
    labels_names=hyperparams['labels_names']
    
    
    # test subject idx, which one is for testing
    test_subject_ids_names = hyperparams['test_subject_ids_names']
    test_subject_ids_str=''
    for ii in test_subject_ids:
        test_subject_ids_str+='_'+str(ii)

    pd_error, pd_NRMSE = estimation_accuracy(predictions,labels,labels_names)

    plot_estimation_accuracy(pd_error, pd_NRMSE)
    


def plot_estimation_accuracy(pd_error, pd_NRMSE):
    # create experiment results folder
    # MAE
    fig=plt.figure(figsize=(10,2))
    style = ['darkgrid', 'dark', 'white', 'whitegrid', 'ticks']
    sns.set_style(style[4],{'grid.color':'k'})
    sns.catplot(data=pd_error,kind='bar', palette="Set3").set(ylabel='Absolute error [deg]')
    #plt.text(2.3,1.05, r"$\theta_{ae}(t)=abs(\hat{\theta}(t)-\theta)(t)$",horizontalalignment='center', fontsize=20)
    test_subject_ids_names = hyperparams['test_subject_ids_names']
    savefig_file=testing_folder+'/sub'+str(test_subject_ids_str)+'_mae.svg'
    plt.savefig(savefig_file)
    
    # NRMSE
    fig=plt.figure(figsize=(10,3))
    sns.catplot(data=pd_NRMSE,kind='bar', palette="Set3").set(ylabel='NRMSE [%]')
    #plt.text(2.3, 2.6, r"$NRMSE=\frac{\sqrt{\sum_{t=0}^{T}{\theta^2_{ae}(t)}/T}}{\theta_{max}-\theta_{min}} \times 100\%$",horizontalalignment='center', fontsize=20)
    savefig_file=testing_folder+'/sub'+str(test_subject_ids_str)+'_nrmse.svg'
    plt.savefig(savefig_file)



    
def estimation_accuracy(estimation, actual, labels_names):
    # Plot the statistical results of the estimation results and errors
    error=abs(estimation-actual)
    pd_error=pd.DataFrame(data=error,columns=labels_names)
    NRMSE=100.0*np.sqrt(pd_error.apply(lambda x: x**2).mean(axis=0).to_frame().transpose())/(actual.max(axis=0)-actual.min(axis=0))
    #*np.ones(pd_error.shape)*100
    pd_NRMSE=pd.DataFrame(data=NRMSE, columns = [col for col in list(pd_error.columns)])

    return pd_error, pd_NRMSE 



'''
Plot the history metrics in training process

'''

def plot_history(history):

    history_dict = history
    print(history_dict.keys())
    plt.plot(history_dict['loss'],'r')
    plt.plot(history_dict['val_loss'],'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(['train loss', 'valid loss'])

    plt.figure()
    plt.plot(history_dict['mae'],'r')
    plt.plot(history_dict['val_mae'],'g')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend(['train mae', 'valid mae'])

    print('Max train and validtion MAE: {:.4f} and {:.4f}'.format(max(history_dict['mae']),max(history_dict['val_mae'])))



'''
plot a curve of estimation and its actual value

Input: pd_labels, pd_predictions


'''

def plot_actual_estimation_curves(pd_labels, pd_predictions, testing_folder=None, fig_save_folder=None,**kwargs):
    """
    Plot the comparison between actual and prediction reslus

    """
    #1) load dataset
    #i) add time and legends to pd_labels and pd_predictions
    if(not isinstance(pd_labels,pd.DataFrame)):
        pd_labels = pd.DataFrame(pd_labels)
        pd_labels = pd.DataFrame(pd_labels)

    pd_labels = copy.deepcopy(pd_labels)
    pd_predictions = copy.deepcopy(pd_predictions)

    Time=np.linspace(0,pd_labels.shape[0]/SAMPLE_FREQUENCY,num=pd_labels.shape[0])
    pd_labels['Time']=Time
    pd_predictions['Time']=Time

    pd_labels['Legends']='Actual'
    pd_predictions['Legends']='Prediction'

    #iii) organize labels and predictions into a pandas dataframe
    pd_labels_predictions=pd.concat([pd_labels,pd_predictions],axis=0)
    pd_labels_predictions=pd_labels_predictions.melt(id_vars=['Time','Legends'],var_name='Variables',value_name='Values')

    #2) plot dataset and save figures

    # plot configuration
    figwidth = 5; figheight = 5
    subplot_left=0.08; subplot_right=0.95; subplot_top=0.9;subplot_bottom=0.1; hspace=0.12; wspace=0.12

    #i) plot estimation results
    g=sns.FacetGrid(data=pd_labels_predictions,col='Variables',hue='Legends',sharey=False)
    g.map_dataframe(sns.lineplot,'Time','Values')
    g.add_legend()
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=hspace,wspace=wspace)
    if(g.ax!=None):
        g.ax.grid(axis='both',which='major')
        if('fig_title' in kwargs.keys()):
            g.ax.set_title(kwargs['fig_title'])
        if('metrics' in kwargs.keys()):
            g.ax.text(0.6, 2,kwargs['metrics'], fontsize=12) #add text
    else:
        [ax.yaxis.grid(axis='both',which='major') for ax in g.axes]
        if('fig_title' in kwargs.keys()):
            [ax.set_title(kwargs['fig_title']) for ax in g.axes]
        if('metrics' in kwargs.keys()):
            [ax.text(0.45, 2, kwargs['metrics'], fontsize=12) for ax in g.axes]#add text


    #ii) save figure
    if testing_folder!=None:
        # whether define the figsave_file
        if('fig_title' in kwargs.keys()):
            figPath = os.path.join(testing_folder, kwargs['fig_title']+".svg")
        else:
            figPath = os.path.join(testing_folder, str(localtimepkg.strftime("%H_%M_%S", localtimepkg.localtime())) + '.svg')

        plt.savefig(figPath)
    
    #iii) to show plot or not
    if('verbose' in kwargs.keys() and kwargs['verbose']==1):
        plt.show()

    plt.close()




'''
Plot statistic atucal and estimation values

'''








def plot_estimation_error(labels,predictions,labels_names,fig_save_folder=None,**kwargs):
    """
    Plot the error between the atual and prediction

    """
    print("Plot the error beteen the actual and prediction results")

    #i) calculate estimation errors statistically: rmse. It is an average value
    pred_error=predictions-labels
    pred_mean=np.mean(pred_error,axis=0)
    pred_std=np.std(pred_error,axis=0)
    pred_rmse=np.sqrt(np.sum(np.power(pred_error,2),axis=0)/pred_error.shape[0])
    pred_rrmse=pred_rmse/np.mean(labels,axis=0)*100.0
    print("mean of ground-truth:",np.mean(labels))
    print("mean: {.2f}, std: {.2f}, RMSE: {.2f}, rRMSE: {.2f} of the errors between estimation and ground truth",pred_mean, pred_std, pred_rmse, pred_rrmse)


    #ii) calculate estimation errors realtime: normalized_absolute_error (nae)= abs(labels-prediction)/labels, along the time, each column indicates a labels
    nae = np.abs(pred_error)#/labels
    pd_nae=pd.DataFrame(data=nae,columns=labels_names);pd_nae['time']=Time
    pd_nae=pd_nae.melt(id_vars=['time'],var_name='GRF error [BW]',value_name='vals')


    #iii) plot absolute error and noramlized error (error-percentage)
    g=sns.FacetGrid(data=pd_nae,col='GRF error [BW]',col_wrap=3,sharey=False)
    g.map_dataframe(sns.lineplot,'time','vals')


    #ii) save figure
    if(fig_save_folder!=None):
        folder_fig = fig_save_folder + "/"
    else:
        folder_fig = "./"

    if not os.path.exists(folder_fig):
        os.makedirs(folder_fig)

    # figure save file
    if('prediction_error_file' in kwargs.keys()):
        figPath = kwargs['prediction_error_file']
    else:
        figPath = folder_fig + str(localtimepkg.strftime("%Y-%m-%d %H:%M:%S", localtimepkg.localtime())) + '_test_mes.svg'

    plt.savefig(figPath)



'''


Test each trained model on a trial of a testing subject

Plot the testing results of combination investigation


'''

def plot_combination_investigation_results(combination_investigation_results, investigation_variable='Sensor configuration', displayed_variables=['r2','r_rmse']):

    #0) load data
    data = get_investigation_metrics(combination_investigation_results, displayed_variables)


    #1) create folder
    combination_investigation_folder = re.search("[\s\S]+(\d)+", combination_investigation_results).group()

    #2) plot statistical results
    figwidth = 13; figheight = 10
    subplot_left=0.06; subplot_right=0.97; subplot_top=0.95; subplot_bottom=0.06

    g = sns.catplot(data=data, x=investigation_variable, y='scores', col='metrics', col_wrap=2, kind='bar', hue='fields', height=3, aspect=0.8, sharey=False)
    g.fig.subplots_adjust(left = subplot_left, right=subplot_right, top=subplot_top, bottom=subplot_bottom, hspace=0.1, wspace=0.1)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    [ax.yaxis.grid(True) for ax in g.axes]
    g.savefig(os.path.join(combination_investigation_folder, "metrics.svg"))





'''
Test each trained model on the testing subjects' all trials and plot them:
    combination_investigation_results can be a list contains training folders or pd dataframe conatins metrics

'''

def plot_model_evaluation_on_unseen_subject(combination_investigation_results, investigation_variable='Sensor configurations', displayed_metrics = ['r2','r_rmse']):

    #0) calculate assessment metrics
    if(re.search('metrics',combination_investigation_results)):
        pd_assessment = pd.read_csv(combination_investigation_results, header=0)
    elif(isinstance(combination_investigation_results,pd.DataFrame)):
        pd_assessment = combination_investigation_results
    else:
        pd_assessment = get_investigation_assessment(combination_investigation_results)
        
    overall_metrics_folder = os.path.dirname(combination_investigation_results)
    # save r2 scores
    pd_assessment.groupby('metrics').get_group('r2').to_csv(os.path.join(overall_metrics_folder,"r2_metrics.csv"))
    pd_assessment.groupby('metrics').get_group('r_rmse').to_csv(os.path.join(overall_metrics_folder,"r_rmse_metrics.csv"))

    # plot statistical results
    # i) plot configuration
    figwidth=13;figheight=10
    subplot_left=0.06; subplot_right=0.97; subplot_top=0.95;subplot_bottom=0.06

    # ii) plot 
    displayed_pd_assessment = pd_assessment[pd_assessment['metrics'].isin(displayed_metrics)]
    g=sns.catplot(data=displayed_pd_assessment, x=investigation_variable, y='scores',col='metrics',col_wrap=2,kind='bar',hue='fields',height=3, aspect=0.8,sharey=False)
    g.fig.subplots_adjust(left=subplot_left,right=subplot_right,top=subplot_top,bottom=subplot_bottom,hspace=0.1, wspace=0.1)
    g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
    [ax.yaxis.grid(True) for ax in g.axes]

    # iii) save plot figure
    g.savefig(os.path.join(overall_metrics_folder,"metrics.svg"))
    
    return pd_assessment



def setup_plot(g, **kwargs):
               
    '''
    set up plot configurations

    '''
    xlabel = 'LSTM units'
    ylabel = 'R2'
    
    if('xlabel' in kwargs.keys()):
        xlabel = kwargs['xlabel']
    if('figtitle' in kwargs.keys()):
        figtitle = kwargs['figtitle']
    if('metrics' in kwargs.keys()):
        text = kwargs['metrics']
        
    if(hasattr(g, 'ax')): # only a subplot
        g.ax.grid(axis='both',which='major')
        g.ax.set_xlabel(xlabel)
        g.ax.set_ylabel('R2')
        if('figtitle' in kwargs.keys()):
            g.ax.set_title(figtitle)
        if('metrics' in kwargs.keys()):
            g.ax.text(0.6, 2,text, fontsize=12) #add text
    elif(hasattr(g, 'axes') and isinstance): # multi subplots
        try:
            iter(g.axes)
            pdb.set_trace()
            [ax.grid(axis='both',which='major') for ax in g.axes]
            [ax.set_xlabel(xlabel) for ax in g.axes]
            [ax.set_ylabel('R2') for ax in g.axes]
            if('figtitle' in kwargs.keys()):
                [ax.set_title(kwargs['figtitle']) for ax in g.axes]
            if('metrics' in kwargs.keys()):
                [ax.text(0.45, 2, kwargs['metrics'], fontsize=12) for ax in g.axes]#add text
        except TypeError: # only an axes
            g.axes.grid(axis='both',which='major')
            g.axes.set_xlabel(xlabel)
            g.axes.set_ylabel('R2')
            g.axes.legend(ncol=3,title='Sensor configurations',loc='lower right')
            
    if(isinstance(g,plt.Axes)):
        g.set_xlabel(xlabel)
        g.set_ylabel('R2')
        g.grid(visible=True, axis='both',which='major')
        g.set_ylim(0.7,1.0)
        g.legend(ncol=3,title='Sensor configurations',loc='lower right')
        g.get_legend().remove()


#---------------------------------- The following function for visualizaton on Papers-----------------------------#



'''
Plot the estimation accuracy related to LSTM units and sensor configurations


'''

def plot_sensorconfig_modelsize_investigation_results(combination_investigation_results, landing_manner='double_legs', estimated_variable='[GRF]', syn_features_label='both', LSTM_unit='all', IMU_number='all', title=None, drop_value=None, metric_fields=['r2']):

    #1) load assessment metrics
    metrics, hue = parse_metrics(combination_investigation_results, 
                                       landing_manner=landing_manner, 
                                       estimated_variable=estimated_variable, 
                                       syn_features_label=syn_features_label,
                                       LSTM_unit=LSTM_unit,
                                       IMU_number=IMU_number,
                                       drop_value=drop_value,
                                       metric_fields=metric_fields)
    #2) plot
    # i) plot configurations
    figsize=(13,6)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(2,4)#13
    gs1.update(hspace=0.25,wspace=0.34,top=0.93,bottom=0.12,left=0.06,right=0.95)
    axs = []
    axs.append(fig.add_subplot(gs1[0, 0:2]))
    axs.append(fig.add_subplot(gs1[0, 2:4]))
    axs.append(fig.add_subplot(gs1[1, 0:2]))
    axs.append(fig.add_subplot(gs1[1, 2:4]))

    #ii) plot colors
    colors =  sns.color_palette("Paired")
    colors = sns.color_palette("YlGnBu")
    #colors = sns.cubehelix_palette(start=.5, rot=-.5)

    #iii) sensor configurations
    single_imu = ['T','S','F','W','C']
    double_imus = ['FS','FT','FW','FC','ST','SW','SC','TW','TC','WC']
    triad_imus = ['FST','FSW','FSC','FTW','FTC','FWC','STW','STC','SWC','TWC']
    quad_imus = ['FSTW','FSTC','FSWC','FTWC','STWC','FSTWC']

    #iv) plotting
    for idx, imu_config in enumerate([single_imu, double_imus, triad_imus, quad_imus]):
        x='LSTM units'; y = 'scores'
        displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
        hue_plot_params = {
            'data': displayed_data,
            'x': x,
            'y': y,
            "hue":  "Sensor configurations",
            "color": colors[idx]
            }
        #pdb.set_trace()
        g = sns.lineplot(ax=axs[idx], **hue_plot_params, sort=True)
        axs[idx].set_xlabel('LSTM units')
        axs[idx].set_ylabel('$R^2$')
        axs[idx].grid(visible=True, axis='both', which='major')
        axs[idx].set_xticks([0, 50, 100, 150, 200])
        axs[idx].legend(ncol= 5+0*len(axs[idx].legend().get_texts()),title='Sensor configurations',loc='best')
    # set title
    fig.suptitle(re.search('[A-Z]+',estimated_variable).group(0) + title)

    # save figures
    return save_figure(os.path.dirname(combination_investigation_results),fig_name='GRF_estimation',fig_format='svg'), metrics





'''
 Plot the estimation accuracy related to LSTM units and sensor configurations

'''

def lineplot_IMU_number_LSTM_unit_accuracy(combination_investigation_results, 
                                                              landing_manner='double_legs', 
                                                              estimated_variable='[GRF]', 
                                                              syn_features_label='both', 
                                                              use_frame_index=True,
                                                              LSTM_unit='all', 
                                                              IMU_number='all',
                                                              title=None,
                                                              y='scores',
                                                              hue=None,
                                                              drop_value=None, 
                                                              metric_fields=['r2']):


    
    #1) load assessment metrics
    metrics = parse_list_investigation_metrics(combination_investigation_results, landing_manner=landing_manner, estimated_variable=estimated_variable, 
                                                syn_features_label=syn_features_label, 
                                                use_frame_index = use_frame_index,
                                                LSTM_unit=LSTM_unit, IMU_number=IMU_number, 
                                                drop_value=drop_value, 
                                                metric_fields=metric_fields,
                                                sort_variable=None)

    #2) plot
    # i) plot configurations
    figwidth =5; figheight=4
    hspace = 0.25; wspace=0.34; top=0.93; bottom=0.12; left=0.14; right=0.95

    '''
    figsize=(figwidth,figheight)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(2,2)#13
    gs1.update(hspace=hspace,wspace=wspace,top=top,bottom=bottom,left=left,right=right)
    axs = []
    axs.append(fig.add_subplot(gs1[0:2, 0:2]))
    '''

    #ii) plot colors
    if hue != None:
        palette = sns.color_palette("Paired")
    else:
        palette = None

    colors = sns.color_palette("YlGnBu")

    #iii) sensor configurations
    idx = 0
    x = 'LSTM units'
    #displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
    hue_plot_params = {
        'data': metrics,
        'x': x,
        'y': y,
        'hue': hue,
        #"color": colors[idx]
        }
    #g = sns.lineplot(ax=axs[idx], **hue_plot_params)
    #g = sns.boxplot(**hue_plot_params)
    g = sns.lmplot(**hue_plot_params,order=2,scatter=False)
    #g = sns.relplot(**hue_plot_params)
    if(isinstance(g,plt.Axes)):
        ax_handle = g
    if(isinstance(g,sns.axisgrid.FacetGrid)):
        ax_handle=g.fig.axes[0]
        g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
        g.fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)
        fig = g.fig
        fig.suptitle(re.search('[A-Z]+', estimated_variable).group(0) + title)

    ax_handle.set_xlabel('LSTM units')
    ax_handle.set_ylabel('$R^2$')
    #ax_handle.set_ylim(0.75,0.95)
    ax_handle.set_xlim(0,200)
    #ax_handle.set_xticks([0, 50, 100, 150, 200])
    ax_handle.set_ylim(0.7, 0.85)
    ax_handle.set_yticks([0.7, 0.75, 0.8, 0.85])
    ax_handle.grid(visible=True, axis='both',which='major')
    if hue!=None:
        ax_handle.legend(ncol=4,title='IMU number',loc='lower right')
        #g.get_legend().remove()


    return save_figure(os.path.dirname(combination_investigation_results[0]),fig_name=title,fig_format='svg'), metrics


'''
 Plot the execution time of each frame related to LSTM units and sensor configurations

'''

def lineplot_IMU_number_LSTM_unit_execution_time(combination_investigation_results, 
                                                              landing_manner='double_legs', 
                                                              estimated_variable='[GRF]', 
                                                              syn_features_label='both', 
                                                              use_frame_index=True,
                                                              LSTM_unit='all', 
                                                              IMU_number='all',
                                                              title=None,
                                                              y='scores',
                                                              hue=None,
                                                              drop_value=None, 
                                                              metric_fields=['r2']):

    #1) load assessment metrics
    metrics = parse_list_investigation_metrics(combination_investigation_results, landing_manner=landing_manner, estimated_variable=estimated_variable, 
                                                syn_features_label=syn_features_label, 
                                                use_frame_index = use_frame_index,
                                                LSTM_unit=LSTM_unit, IMU_number=IMU_number, 
                                                drop_value=drop_value, 
                                                metric_fields=metric_fields,
                                                sort_variable=None)

    #2) plot
    # i) plot configurations
    figwidth =5; figheight=4
    hspace = 0.25; wspace=0.34; top=0.93; bottom=0.12; left=0.14; right=0.95

    colors = sns.color_palette("YlGnBu")

    #iii) sensor configurations
    idx = 0
    x = 'LSTM units'
    #displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
    hue_plot_params = {
        'data': metrics,
        'x': x,
        'y': y
        #"color": colors[idx]
        }
    #g = sns.lineplot(ax=axs[idx], **hue_plot_params)
    #g = sns.boxplot(**hue_plot_params)
    g = sns.lmplot(**hue_plot_params,order=1,scatter=False)
    #g = sns.relplot(**hue_plot_params)
    if(isinstance(g,plt.Axes)):
        ax_handle = g
    if(isinstance(g,sns.axisgrid.FacetGrid)):
        ax_handle=g.fig.axes[0]
        g.fig.set_figwidth(figwidth); g.fig.set_figheight(figheight)
        g.fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)
        fig = g.fig
        fig.suptitle(re.search('[A-Z]+', estimated_variable).group(0) + title)

    ax_handle.set_xlabel('LSTM units')
    ax_handle.set_ylabel('Execution time [ms]')
    #ax_handle.set_ylim(0.75,0.95)
    ax_handle.set_xlim(0,200)
    #ax_handle.set_xticks([0, 50, 100, 150, 200])
    #ax_handle.set_ylim(9.2, 10.2)
    #ax_handle.set_yticks([9.2,9.4,9.6,9.8,10.0,10.2])
    ax_handle.grid(visible=True, axis='both',which='major')
    if hue!=None:
        ax_handle.legend(ncol=4,title='IMU number',loc='lower right')
        #g.get_legend().remove()

    if(not isinstance(combination_investigation_results, list)): # convert into a list
        combination_investigation_results = [combination_investigation_results]
    return save_figure(os.path.dirname(combination_investigation_results[0]),fig_name=title,fig_format='svg'), metrics








'''
Plot the estimation accuracy related to sensor configurations

'''
def plot_sensorconfig_investigation_results(combination_investigation_results, landing_manner='both', estimated_variable='both', 
                                             syn_features_label='both', LSTM_unit='all', IMU_number='all', title=None, drop_value=None, metric_fields=['r2'], hue=None):
    #1) parase data
    metrics = parse_metrics(combination_investigation_results, 
                                        landing_manner=landing_manner, 
                                        estimated_variable=estimated_variable, 
                                        syn_features_label=syn_features_label,
                                        LSTM_unit=LSTM_unit,
                                        IMU_number=IMU_number,
                                        drop_value=drop_value,
                                        metric_fields=metric_fields)


    #2) plot
    # i) plot configurations
    figsize=(13,6)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(2,4)#13
    gs1.update(hspace=0.25,wspace=0.34,top=0.93,bottom=0.12,left=0.06,right=0.95)
    axs = []
    axs.append(fig.add_subplot(gs1[0, 0:2]))
    axs.append(fig.add_subplot(gs1[0, 2:4]))
    axs.append(fig.add_subplot(gs1[1, 0:2]))
    axs.append(fig.add_subplot(gs1[1, 2:4]))

    #ii) plot colors
    if hue != None:
        palette =  sns.color_palette("Paired")
    else:
        palette = None

    colors = sns.color_palette("YlGnBu")
    #colors = sns.cubehelix_palette(start=.5, rot=-.5)

    #iii) sensor configurations
    single_imu = ['T','S','F','W','C']
    double_imus = ['FS','FT','FW','FC','ST','SW','SC','TW','TC','WC']
    triad_imus = ['FST','FSW','FSC','FTW','FTC','FWC','STW','STC','SWC','TWC']
    quad_imus = ['FSTW','FSTC','FSWC','FTWC','STWC','FSTWC']
    all_imu_config = [single_imu, double_imus, triad_imus, quad_imus]
    if(IMU_number=='all'):
        display_imu_list = all_imu_config
    else:
        display_imu_list = [all_imu_config[i-1] for i in IMU_number]

    for idx, imu_config in enumerate(display_imu_list):
        x='Sensor configurations'; y = 'scores'
        displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
        hue_plot_params = {
            'data': displayed_data,
            'x': x,
            'y': y,
            'hue': hue,
            "showfliers": False,
            "palette": palette,
            "color": colors[idx]
        }
        g = sns.boxplot(ax=axs[idx], **hue_plot_params)
        g.set_xlabel('Sensor configurations')
        g.set_ylabel('$R^2$')
        g.set_ylim(0.6,1.0)
        g.set_yticks([0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
        g.grid(visible=True, axis='both',which='major')
        if hue!=None:
            g.legend(ncol=3,title='Event synchronization',loc='lower right')
            #g.get_legend().remove()

    fig.suptitle(re.search('[A-Z]+',estimated_variable).group(0)+title)

    return save_figure(os.path.dirname(combination_investigation_results),fig_name=title,fig_format='svg'), metrics



'''
plot ensemble curves of the actual and estimattion

'''

def plot_statistic_actual_estimation_curves(list_training_testing_folders, list_selections, **kwargs):

    # 1) loading test data 
    multi_test_results = get_multi_models_test_results(list_training_testing_folders, list_selections)

    # i) plot configurations
    figsize=(7,7)
    sns.set(font_scale=1.15,style='whitegrid')
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(2,4)#13
    gs1.update(hspace=0.25,wspace=0.34,top=0.93,bottom=0.12,left=0.06,right=0.95)
    axs = []
    axs.append(fig.add_subplot(gs1[0, 0:2]))
    axs.append(fig.add_subplot(gs1[0, 2:4]))
    axs.append(fig.add_subplot(gs1[1, 0:2]))
    axs.append(fig.add_subplot(gs1[1, 2:4]))

    hue = None
    if hue != None:
        palette = sns.color_palette("Paired")
    else:
        palette = None
    colors = sns.color_palette("YlGnBu")

    for idx, estimation_values in enumerate(multi_test_results):
        x=None; y = None
        hue_plot_params = {
            'data': estimation_values,
            'x': x,
            'y': y,
            'hue': None,
            "palette": palette,
            "color": colors[idx]
            }
        g = sns.lineplot(ax=axs[idx], **hue_plot_params)
        g.set_xlabel('Time [s]')
        g.set_xlim(0, 0.8)
        g.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        if idx == 0:
            g.set_ylim(-0.1, 2)
            g.set_yticks([0, 1, 2])
        elif idx==1:
            g.set_ylim(-0.21, 4)
            g.set_yticks([0, 1, 2, 3, 4])
        elif idx ==2:
            g.set_ylim(-0.2, 4)
            g.set_yticks([0, 1, 2, 3, 4])
        else:
            g.set_ylim(-0.25, 5)
            g.set_yticks([0, 1, 2, 3, 4, 5])
            
        '''
        if('legends' in display_configs):
            g.legend(ncol=1,title=None,loc='upper right',labels=display_configs['legends'][idx])
        if('ylabel' in display_configs):
            g.set_ylabel(display_configs['ylabel'][idx])
        if('subplot_titles' in display_configs):
            g.set_title(label=display_configs['subplot_titles'][idx])
        '''
        g.grid(visible=True, axis='both',which='major')
        
    
    
    return save_figure(os.path.dirname(list_training_testing_folders[0]),fig_format='svg'), multi_test_results


'''
plot overall figures

'''

def boxplot_IMU_number_accuracy(combination_investigation_results, landing_manner='both', estimated_variable='both', syn_features_label='both', use_frame_index='both',LSTM_unit='all', IMU_number='all',title=None, drop_value=None, metric_fields=['r2'], hue=None):

    #1) load assessment metrics
    if(not isinstance(combination_investigation_results,list)):
        combination_investigation_results = [combination_investigation_results]

    list_metrics=[]
    for combination_investigation_result in combination_investigation_results:
        list_metrics.append(parse_metrics(combination_investigation_result, 
                                             landing_manner=landing_manner, 
                                             estimated_variable=estimated_variable, 
                                             syn_features_label=syn_features_label,
                                             use_frame_index = use_frame_index,
                                             LSTM_unit=LSTM_unit,
                                             IMU_number = IMU_number,
                                             drop_value=drop_value,
                                             metric_fields=metric_fields
                                            ))

    metrics=pd.concat(list_metrics,axis=0)
    metrics['IMU number'] = metrics['IMU number'].astype(str)


    #2) plot
    # i) plot configurations
    figsize=(5,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(2,2)#13
    gs1.update(hspace=0.25,wspace=0.34,top=0.95,bottom=0.11,left=0.13,right=0.95)
    axs = []
    axs.append(fig.add_subplot(gs1[0:2, 0:2]))

    #ii) plot colors
    if hue != None:
        palette =  sns.color_palette("Paired")
    else:
        palette = None

    sns.set(font_scale=1.15,style='whitegrid')
    states_palette = sns.color_palette("YlGnBu", n_colors=8)
    colors = sns.color_palette("YlGnBu") 
    #iii) sensor configurations
    
    idx=0
    x='IMU number'; y = 'scores'
    #displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
    hue_plot_params = {
        'data': metrics,
        'x': x,
        'y': y,
        'hue': hue,
        "showfliers": False,
        "showmeans": True,
        "color": colors[idx],
        "palette": states_palette
        }


    g = sns.boxplot(ax=axs[idx], **hue_plot_params)
    g.set_xlabel('Number of IMUs')
    g.set_ylabel('$R^2$')
    g.set_ylim(0.4,1.0)
    g.set_yticks([0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    g.grid(visible=True, axis='both',which='major')
    if hue!=None:
        g.legend(ncol=3,title='Event-based alignment',loc='lower right')
        #g.get_legend().remove()

    #fig.suptitle(re.search('[A-Z]+',estimated_variable).group(0)+title)


    # statistical test
    #test_method="Mann-Whitney"
    test_method="t-test_ind"
    pairs = (
        [('1','2'),('2','3'),('3','4'),('4','5'),('5','6'),('6','7'),('7','8')]
    )

    annotator=Annotator(g,pairs=pairs,**hue_plot_params)
    annotator.configure(test=test_method, text_format='star', loc='outside')
    annotator.apply_and_annotate()

    return save_figure(os.path.dirname(combination_investigation_results[0]),fig_name=title,fig_format='svg'), metrics




def boxplot_single_imu_estimation_accuracy(combination_investigation_results):

    landing_manner= "double_legs"
    metrics = parse_metrics(
        combination_investigation_results[0],
        landing_manner = landing_manner,
        estimated_variable = 'GRF',
        syn_features_label = False,
        drop_value = 0.0,
        IMU_number = [1]
    )
    
    metrics['Sensor configurations'] = metrics['Sensor configurations'].replace(['F'],'R_F')
    metrics['Sensor configurations'] = metrics['Sensor configurations'].replace(['S'],'R_S')
    metrics['Sensor configurations'] = metrics['Sensor configurations'].replace(['T'],'R_T')
    
    landing_manner= "double_legs"
    additional_metrics = parse_metrics(
        combination_investigation_results[1],
        landing_manner = landing_manner,
        estimated_variable = 'GRF',
        syn_features_label = False,
        drop_value = 0.0,
        IMU_number = [1]
    )
    
    additional_metrics['Sensor configurations'] = 'L_' + additional_metrics['Sensor configurations']
    
    all_metrics = pd.concat([metrics,additional_metrics],axis=0)
    
    
    imu_config = ['R_T','R_S','R_F','W','C','L_F','L_S','L_T']
    
    displayed_data = all_metrics.loc[all_metrics['Sensor configurations'].isin(imu_config)]
    sort_variable='average scores'
    displayed_data[sort_variable] = displayed_data[sort_variable].astype('float64')
    displayed_data.sort_values(by=[sort_variable], ascending=True, inplace=True)
    
    
    sns.set(font_scale=1.15,style='whitegrid')
    colors = sns.color_palette("YlGnBu") 
    #colors = sns.cubehelix_palette(start=.5, rot=-.5) 
    
    states_palette = sns.color_palette("YlGnBu", n_colors=8)
    x='Sensor configurations'
    y='scores'
    
    
    hue_plot_params = {
    'data': displayed_data,
    'x': x,
    'y': y,
    'showmeans': True,
    "showfliers":False,
    "palette": states_palette
    }
    
    g = sns.boxplot( **hue_plot_params)
    g.set_ylabel('$R^2$')
    g.set_xlabel('IMU locations')
    g.grid(visible=True, axis='both',which='major')
    g.set_ylim([0.3,1.0])
    g.set_yticks([0.6,0.7,0.8,0.9,1.0])
    
    # statistical test
    #test_method="Mann-Whitney"
    test_method="t-test_ind"
    
    pairs = (
    [('L_S','R_F'),('R_F','W'),('W','L_F'),('L_F','L_T'),('L_T','R_T'),('R_T','R_S'),('R_S','C')]
    )
    
    annotator = Annotator(g,pairs=pairs,**hue_plot_params)
    annotator.configure(test=test_method, text_format='star', loc='outside')
    annotator.apply_and_annotate()
    
    
    save_figure(os.path.dirname(combination_investigation_results[1]),'single_IMU',fig_format='svg')


'''
P6, compare different models

'''
def plot_models_accuracy(combination_investigation_results, title=None, metric_fields=['r2'], hue=None, statannotation_flag=False, **kwargs):

    #1) load assessment metrics
    if(not isinstance(combination_investigation_results,list)):
        combination_investigation_results = [combination_investigation_results]

    list_metrics=[]
    for combination_investigation_result in combination_investigation_results:
        list_metrics.append(parse_metrics(combination_investigation_result, 
                                             metric_fields=metric_fields, **kwargs
                                            ))

    metrics=pd.concat(list_metrics,axis=0)

    #2) plot
    # i) plot configurations
    figsize=(5,4)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(2,2)#13
    gs1.update(hspace=0.25,wspace=0.34,top=0.95,bottom=0.11,left=0.13,right=0.95)
    axs = []
    axs.append(fig.add_subplot(gs1[0:2, 0:2]))

    #ii) plot colors
    if hue != None:
        palette =  sns.color_palette("Paired")
    else:
        palette = None

    sns.set(font_scale=1.15,style='whitegrid')
    states_palette = sns.color_palette("YlGnBu", n_colors=8)
    colors = sns.color_palette("YlGnBu") 

    #iii) sensor configurations
    idx=0
    x='model_selection'; y = 'r2'
    #displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
    plot_params = {
        'data': metrics,
        'x': x,
        'y': y,
        'hue': hue,
        "showfliers": False,
        "showmeans": True,
        "color": colors[idx],
        "palette": states_palette
        }
    if('plot_params' in kwargs.keys()):
        for key, value in kwargs['plot_params'].items():
            plot_params[key] = value
    pdb.set_trace()
    g = sns.boxplot(ax=axs[idx], **plot_params)
    g.set_xlabel('Models')
    g.set_ylabel('$R^2$')
    g.set_ylim(0.4,1.0)
    g.set_yticks([0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    g.grid(visible=True, axis='both',which='major')
    if hue!=None:
        g.legend(ncol=3,title='Event-based alignment',loc='lower right')
        #g.get_legend().remove()
    #fig.suptitle(re.search('[A-Z]+',estimated_variable).group(0)+title)


    # statistical test
    if(statannotation_flag):
        test_method="Mann-Whitney"
        #test_method="t-test_ind"
        pairs = (
            [('baseline','DANN')]
        )
        pairs=kwargs['test_paris']

        annotator=Annotator(g,pairs=pairs,**plot_params)
        annotator.configure(test=test_method, text_format='star', loc='outside')
        annotator.apply_and_annotate()

    return save_figure(os.path.dirname(combination_investigation_results[0]),fig_name=title, fig_format='svg'), metrics



'''
P6 plot ensemble curves of the actual and estimattion

'''

def p6plot_statistic_actual_estimation_curves(list_training_testing_folders, list_selections=None, figsize=(15,12), col_wrap=3,  save_fig=False, save_folder_index=0, save_format='.png', title='curves', font_scale=1, **kwargs):
    
    #1) get testing results: estimation and ground truth
    multi_test_results = get_multi_models_test_results(list_training_testing_folders, list_selections)
    
    #2) set figures
    row_num = math.ceil(len(list_training_testing_folders)/col_wrap)
    fig = plt.figure(figsize=figsize,constrained_layout=False)
    gs1 = gridspec.GridSpec(row_num,col_wrap)
    gs1.update(hspace=0.25,wspace=0.2,top=0.93,bottom=0.12,left=0.06,right=0.95)
    axs = []
    for row_idx in range(row_num):
        for col_idx in range(col_wrap):
            axs.append(fig.add_subplot(gs1[row_idx, col_idx:col_idx+1]))
            
    
    sns.set(font_scale=font_scale,style='whitegrid')
    palette = sns.color_palette("Paired")
    colors = sns.color_palette("YlGnBu")
    if 'ylabels' not in kwargs.keys():
        ylabels = len(list_training_testing_folders)*['KEM']
    else:
        ylabels = kwargs['ylabels']

    if 'titles' not in kwargs.keys():
        titles = [str(os.path.basename(os.path.dirname(a_model_results_foler))) for a_model_results_foler in list_training_testing_folders]
    else:
        titles = kwargs['titles']

    for idx, (ylabel, estimation_values) in enumerate(zip(ylabels, multi_test_results)):
        estimation_values.rename(columns={'Actual R_KNEE_MOMENT_X': 'Ground truth', 'Estimated R_KNEE_MOMENT_X': 'Estimation'}, inplace=True)
        x=None; y = None
        hue_plot_params = {
            'data': estimation_values,
            'x': x,
            'y': y,
            'hue': None,
            #"palette": palette,
            #"color": colors[idx]
            }
        g = sns.lineplot(ax=axs[idx], **hue_plot_params)
        g.set_xlabel('Time [s]')
        g.set_ylabel(ylabel)
        g.set_title(titles[idx])

        axs[idx].spines.right.set_visible(False)
        axs[idx].spines.top.set_visible(False)

        if('xticks' in kwargs.keys()):
            xticks = kwargs['xticks']
            g.set_xticks(xticks)

        if('ylim' in kwargs.keys()):
            g.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

        if('yticks' in kwargs.keys()):
            yticks = kwargs['yticks']
            g.set_yticks(yticks)


        g.set_xlim(0, 0.8)
        g.grid(visible=True, axis='both',which='major')


    if(save_fig):
        if('fig_path' in kwargs.keys()):
            fig_results = save_figure(os.path.dirname(list_training_testing_folders[save_folder_index]),fig_path=kwargs['fig_path'],fig_name=title,fig_format=save_format)
        else:
            fig_results = save_figure(os.path.dirname(list_training_testing_folders[save_folder_index]),fig_name=title,fig_format=save_format)
        print(fig_results)
    else:
        fig_results = 0
        
    return fig_results, multi_test_results


def p6plot_model_accuracy(combination_investigation_metrics,filters={}, ttest=False, save_fig=False, figsize=(14,7), save_format='.svg', font_scale=1, save_folder_index=0, title='model comparison', plot_type='barplot',**kwargs):
    
    # load metrics
    metrics = parse_list_investigation_metrics(combination_investigation_metrics,**filters)


    if('replace_columns' in kwargs.keys()):
        replace_columns = kwargs['replace_columns']
    else:
        replace_columns = {'alias_name':'Model name'}
    if('replace_values' in kwargs.keys()):
        replace_values = kwargs['replace_values']
    else:
        replace_values = {}

    metrics.rename(columns =replace_columns, inplace = True)
    metrics.replace(replace_values, inplace=True)

    # seaborn setup
    sns.set(font_scale=font_scale,style='whitegrid')

    # plot config
    if('hue' in kwargs.keys()):
        hue = kwargs['hue']
    else:
        hue = None

    if('x' in kwargs.keys()):
        x = kwargs['x']
    else:
        x = 'Model name' 

    y = 'r2'
    #displayed_data = metrics.loc[metrics['Sensor configurations'].isin(imu_config)]
    hue_plot_params = {
        'data': metrics,
        'x': x,
        'y': y,
        'hue':hue,
    }

    if(plot_type=='barplot'):
        g = sns.barplot(**hue_plot_params)
    if(plot_type=='boxplot'):
        hue_plot_params["showfliers"]= False
        hue_plot_params["showmeans"] =  True
        g = sns.boxplot(**hue_plot_params)

    if('xticks' in kwargs.keys()):
        xticks = kwargs['xticks']
        g.set_xticks(xticks)

    if('yticks' in kwargs.keys()):
        yticks = kwargs['yticks']
        g.set_yticks(yticks)
        g.set_ylim(yticks[0]-0.1, yticks[-1]+0.1)


    g.set_xlabel(x)
    g.set_ylabel('$R^2$')
    #g.set_ylim(0.4,1.0)
    g.grid(visible=True, axis='both',which='major')

    # significantly test
    if ttest:
        test_method="t-test_ind"
        if('test_pairs' in kwargs.keys()):
            pairs = kwargs['test_pairs']
        annotator=Annotator(g, pairs=pairs,**hue_plot_params)
        annotator.configure(test=test_method, text_format='star', loc='inside')
        annotator.apply_and_annotate()


    fig=g.get_figure()
    fig.set_figwidth(figsize[0]); fig.set_figheight(figsize[1])
    if(save_fig):
        if('fig_path' in kwargs.keys()):
            fig_results = save_figure(os.path.dirname(combination_investigation_metrics[save_folder_index]),fig_path=kwargs['fig_path'],fig_name=title,fig_format=save_format)
        else:
            fig_results = save_figure(os.path.dirname(combination_investigation_metrics[save_folder_index]),fig_name=title,fig_format=save_format)
        print(fig_results)
    else:
        fig_results = 0

        
    return fig_results


if __name__ == '__main__':

    if False:
        # compare baseline and imu augmentation
        combination_investigation_results = [
            os.path.join(RESULTS_PATH, "training_testing","investigation_baseline_v1",str(trial_idx)+"trials",str(sub_idx)+"sub","testing_result_folders.txt") for sub_idx in range(5,11,1) for trial_idx in range(5,16,5)
                                            ]+ [
            os.path.join(RESULTS_PATH, "training_testing","investigation_imu_augment_v1",str(trial_idx)+"trials",str(sub_idx)+"sub","testing_result_folders.txt") for sub_idx in range(5,11,1) for trial_idx in range(5,16,5)
                                            ]
        
        #metrics = get_list_investigation_metrics(combination_investigation_results)
        combination_investigation_metrics = [os.path.join(os.path.dirname(folder),"metrics.csv") for folder in combination_investigation_results]
        
        #subs = list(set(metrics['alias_name']))
        #replace_values = {sub: int(sub.split('v')[1])-1 for sub in subs}
        replace_values = {}
        replace_values.update({'baseline': 'Measured dataset', 'imu_augment': 'Augmented dataset'})
        print(replace_values)
        replace_columns = {'subject_num': 'Train subject number', 'trial_num': 'Trial number', 'model_selection': 'Dataset'}
        
        plot_config={
            "save_fig": True, "save_format":"svg", "save_folder_index": 0,
            'figsize':(8, 5),
            "hue": 'Dataset',
            'replace_values': replace_values,
            'replace_columns': replace_columns,
            'x': 'Train subject number',
            'title': 'baseline',
            'yticks': (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            #'plot_title': 'baseline',
            'font_scale': 1.0,
            'plot_type': 'barplot',
            'ttest':True,
            'test_pairs':(
                [(1,'Measured dataset'), (1,'Augmented dataset')],
                [(2,'Measured dataset'), (2,'Augmented dataset')],
                [(3,'Measured dataset'), (3,'Augmented dataset')],
                [(4,'Measured dataset'), (4,'Augmented dataset')],
                [(5,'Measured dataset'), (5,'Augmented dataset')],
                [(6,'Measured dataset'), (6,'Augmented dataset')],
                         )
        }

        filters={'drop_value':0.0,'sort_variable':'r2'}
        p6plot_model_accuracy(combination_investigation_metrics, **plot_config)
        plt.show()


        exit()



        combination_investigation_results = [
            os.path.join(RESULTS_PATH, "training_testing","investigation_imu_augment_v1",str(trial_idx)+"trials",str(sub_idx)+"sub","testing_result_folders.txt") for sub_idx in [5] for trial_idx in [5]
        ] +[
            os.path.join(RESULTS_PATH, "training_testing","investigation_imu_augment_v1",str(trial_idx)+"trials",str(sub_idx)+"sub","testing_result_folders.txt") for sub_idx in [10] for trial_idx in [25]
        ]



        config = {
            'xticks':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'yticks':[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            'ylim':[-1.0, 3.0],
            'figsize': (8,3),
            'ylabels': 5*['Normalized KFM'],
            'titles': ['A subject with five trials', 'Six subjects with 25 trials'],
            'font_scale':1.2, 
            'save_fig': True,
            'save_format': '.svg'
        }


        figpath, multi_model_results = p6plot_statistic_actual_estimation_curves(combination_investigation_results, col_wrap=2, **config)

        pdb.set_trace


        # baseline
        fliters={'drop_value':0.0,'sort_variable':'r2'}
        combination_investigation_results = [
            os.path.join(RESULTS_PATH, "training_testing","investigation_5trials_baseline","5trials_baseline_v"+str(idx),"testing_result_folders.txt") for idx in range(2,7)
        ]
        pdb.set_trace()

        #metrics = get_list_investigation_metrics(combination_investigation_results)
        combination_investigation_metrics = [os.path.join(os.path.dirname(folder),"metrics.csv") for folder in combination_investigation_results]
        metrics = get_list_investigation_metrics(combination_investigation_metrics)

        subs = list(set(metrics['alias_name']))
        replace_values = {sub: int(sub.split('v')[1])-1 for sub in subs}
        print(replace_values)

        plot_config={
        "save_fig": True, "save_format":"jpg", "save_folder_index": 0,
        "hue": None,
        'replace_values': replace_values,
        'replace_columns': {'alias_name': 'Train subject number'},
        'x': 'Train subject number',
        'title': 'baseline',
        'plot_title': 'baseline'
        }

        p6plot_model_accuracy(combination_investigation_metrics,fliters, ttest=False, **plot_config)
        

        pdb.set_trace()

    '''------------------------P5  --------------'''
    if False: # calculate metrics
        combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/Four_variable_optimal_imu_config/KEM_single_leg/testing_result_folders.txt"
        metrics = get_list_investigation_metrics(combination_investigation_results)
        combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/Four_variable_optimal_imu_config/vGRF_single_leg/testing_result_folders.txt"
        metrics = get_list_investigation_metrics(combination_investigation_results)

        combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/Four_variable_optimal_imu_config/vGRF_double_legs/testing_result_folders.txt"
        metrics = get_list_investigation_metrics(combination_investigation_results)

        combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/Four_variable_optimal_imu_config/KEM_double_legs/testing_result_folders.txt"
        metrics = get_list_investigation_metrics(combination_investigation_results)


    if False: # Fig. 6 IMU and LSTM line
        combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/3_collected_modeling/additional_imu_all_imu_all_lstm_double_GRF/testing_result_folders.txt"
        t1 = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/metrics.csv"
        t2 = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/6_7_8_imu/6_7_8_imu_all_lstm/metrics.csv"
        t3 = [
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/1_imu/1_imu_125_175/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/1_imu/1_imu_25_lstm/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/2_imu/2_imu_25_lstm/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/2_imu/2_imu_75/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/3_imu/3_imu_25_lstm_units/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/3_imu/3_imu_75/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/3_imu/3_imu_125_175/metrics.csv",

            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/4_imu/4_5_imu_125/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/4_imu/4_imu_25_lstm_units/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/4_imu/4_imu_75/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/4_imu/4_imu_175/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/5_imu/5_imu_25/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/5_imu/5_imu_75/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/5_imu/5_imu_125/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/5_imu/5_imu_175/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/6_7_8_imu/6_7_8_imu_25_lstm/metrics.csv",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/6_7_8_imu/75_125_175/metrics.csv"
        ]
        combination_investigation_results = [t1, t2] +t3
        fig_path,metrics = lineplot_IMU_number_LSTM_unit_accuracy(combination_investigation_results,
                                                                  estimated_variable = 'GRF',
                                                                  landing_manner='double_legs',
                                                                  syn_features_label = False,
                                                                  title = ' estimation in double-leg drop landing',
                                                                  drop_value = 0.0,
                                                                  hue = 'IMU number'
                                                                 )


    
    if False: # Fig. 7 execution time
        # combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/execution_time/metrics.csv"
        # combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/execution_time/testing_result_folders.txt"
        fig_path,metrics = lineplot_IMU_number_LSTM_unit_execution_time(combination_investigation_results,
                                                                        estimated_variable = 'GRF',
                                                                        landing_manner='double_legs',
                                                                        syn_features_label = False,
                                                                        title = ' estimation in double-leg drop landing',
                                                                        drop_value = 0.0,
                                                                        y='execution_time'
                                                                        #hue = 'IMU number'
                                                                       )


    if False: #  Fig. 8 
        #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/3_collected_modeling/additional_imu_all_imu_all_lstm_double_GRF/8_imu_all_lstm_units/testing_result_folders.txt"
        #combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/3_collected_modeling/additional_imu_all_imu_all_lstm_double_GRF/metrics.csv"
        fig_path, r2 = boxplot_IMU_number_accuracy(combination_investigation_results,
                                                   landing_manner='double_legs', 
                                                   estimated_variable='GRF',
                                                   syn_features_label=False,
                                                   title=' estimation in double-leg drop landing',
                                                   LSTM_unit=[100], drop_value=0.0)



    if False: # Fig. 9
        t_right = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/2_collected_full_cv/r2_metrics.csv"
        t_left = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/4_collected_sensor_lstm/additional_single_imu/metrics.csv"
        combination_investigation_results = [t_right, t_left]
        boxplot_single_imu_estimation_accuracy(combination_investigation_results)



    if False: #Fig. 5
        combination_investigation_results = "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/1_collected_data/testing_result_folders.txt"
        list_combination_investigation_results = [
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/2_collected_full_cv/4_imu/testing_result_folders.txt",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/20_doubleKFM_collected_full_cv/5_imu/testing_result_folders.txt",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/21_singleGRF_collected_full_cv/5_imu/testing_result_folders.txt",
            "/media/sun/DATA/Drop_landing_workspace/suntao/Results/Experiment_results/training_testing/22_singleKFM_collected_full_cv/5_imu/testing_result_folders.txt"
        ]

        list_selections = [
            {'sensor_configurations': ['STWC']},
            {'estimated_variable': 'KFM'},
            {'estimated_variable': 'GRF'},
            {'estimated_variable': 'KFM' }
        ]

        fig_path, metrics = plot_statistic_actual_estimation_curves(list_combination_investigation_results,
                                                                list_selections)


    # P6 visualization
    if True:
        combination_investigation_results = [
                #os.path.join(RESULTS_PATH, "training_testing/baseline_v5/25trials/15sub/testing_result_folders.txt"),
                #os.path.join(RESULTS_PATH, "training_testing/augmentation_v5/25trials/15sub/testing_result_folders.txt")
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_1/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_2/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_3/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_4/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_5/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_6/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_7/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_8/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_9/25trials/15sub/testing_result_folders.txt"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_10/25trials/15sub/testing_result_folders.txt")
                ]

        combination_investigation_results = [os.path.join(RESULTS_PATH, "training_testing/baseline_v8_5/25trials/"+str(sub)+"sub/testing_result_folders.txt") for sub in range(5,16)]
        #metrics = get_list_investigation_metrics(combination_investigation_results)
        
        combination_investigation_results = [
                #os.path.join(RESULTS_PATH, "training_testing/baseline_v6_10/25trials/15sub/metrics.csv"),
                #os.path.join(RESULTS_PATH, "training_testing/augmentation_v5/25trials/15sub/metrics.csv")
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_1/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_2/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_3/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_4/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_5/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_6/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_7/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_8/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_9/25trials/15sub/metrics.csv"),
                os.path.join(RESULTS_PATH, "training_testing/augmentation_v6_10/25trials/15sub/metrics.csv")
                ]
        combination_investigation_results = [os.path.join(RESULTS_PATH, "training_testing/baseline_v8_5/25trials/"+str(sub)+"sub/metrics.csv") for sub in range(5,16)]
        
        combination_investigation_results = [
            os.path.join(RESULTS_PATH, "training_testing", "augmentation_v17",str(rot_id)+'rotid', str(sub_num)+"sub", str(trial_num)+"tri",  
                         "testing_result_folders.txt") for sub_num in range(15,16,1) for trial_num in range(25, 26,5) for rot_id in [50, 70, 90, 110]
        ]+[
            os.path.join(RESULTS_PATH, "training_testing", "augmentation_v16",str(rot_id)+'rotid', str(sub_num)+"sub", str(trial_num)+"tri",  
                         "testing_result_folders.txt") for sub_num in range(15,16,1) for trial_num in range(25, 26,5) for rot_id in [10,30]
        ]

        combination_investigation_metrics = [os.path.join(os.path.dirname(folder), "metrics.csv") for folder in combination_investigation_results]
        fig_path, r2 = plot_models_accuracy(combination_investigation_results,plot_params={'x':'config_name','y':'r2'})
        #fig_path, r2 = plot_models_accuracy(combination_investigation_results, hue="relative_result_folder", title= 'test', statannotation_flag=False)

