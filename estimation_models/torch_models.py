#!/usr/bin/env python
'''
 Import necessary packages

'''
import torch
print("pytorch version:",torch.__version__)
import numpy as np
import pdb
import os
import pandas as pd
import yaml
import h5py
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


'''
Model_V1 definition
'''

class MLNNBackbone(nn.Module):
    def __init__(self, n_input=49, seg_len=80, n_output=1, lstm_unit=60):
        super(MLNNBackbone, self).__init__()
        self.layer_input = nn.LSTM(input_size=n_input, hidden_size = lstm_unit, num_layers=1, dropout=0.2, bidirectional=True,batch_first=True)
        self.linear_1 = nn.Linear(60,60)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.linear_2 = nn.Linear(60,30)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        self.linear_3 = nn.Linear(30,n_output)
        self.relu_3 = nn.Tanh()

        self.lstm_unit = lstm_unit
        self.seq_len = seq_len
        self._feature_dim = n_input

    def forward(self, x): # input dim = [batch_size, seq_en, features_len]
        x, (hidden, c) = self.layer_input(x) # output dim  = [batch, seq_len, model dim]
        hidden = torch.cat([hidden[:,-2,:], hidden[:,-1,:]], dim=1) # hidden, [batch_size, model_dim *2]
        x = self.linear_1(hidden) # linear input dim =[batch, -1]
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.linear_3(x)
        x = self.relu_3(x)
        return x

    def output_num(self):
        return self._feature_dim



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


'''
Model training

'''
def train_model(model, hyperparams, train_set, valid_set, training_mode='Integrative_way'):
    # specify a training session
    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    
    # crerate train results folder
    training_folder = pro_rd.create_training_files(hyperparams=hyperparams)

    # define callbacks
    callbacks = my_callbacks(training_folder)

    # save model weights
    #model.save_weights(checkpoint_path.format(epoch=0))

    # register tensorboard writer
    sensorboard_file = os.path.join(training_folder,'tensorboard')
    if(os.path.exists(sensorboard_file)==False):
        os.makedirs(sensorboard_file)
    summary_writer = tf.summary.create_file_writer(sensorboard_file)
    
    # optimizer
    #optimizer = tf.keras.optimizers.SGD(learning_rate=hyperparams['learning_rate'], momentum=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    
    """ Integrated mode   """
    if training_mode=='Integrative_way':
        # compile model
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=optimizer,
                      metrics=["mae"]
                     )


        # fit model
        history = model.fit(train_set, 
                            epochs=hyperparams['epochs'],
                            validation_data = valid_set,
                            callbacks = callbacks
                           )
        #model.summary()
    """ Specified mode   """
    if training_mode=='Manual_way':
        tf.summary.trace_on(profiler=True) # 开启trace
        for batch_idx, (X,y_true) in enumerate(train_set): 
            with tf.GradientTape() as tape:
                y_pred=model(X)
                loss=tf.reduce_mean(tf.square(y_pred-y_true))
                # summary writer
                with summary_writer.as_default():
                    tf.summary.scalar('loss',loss,step=batch_idx)
            # calculate grads
            grads=tape.gradient(loss, model.variables)
            # update params
            optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
        # summary trace
        history_dict={"loss":'none'}
        with summary_writer.as_default():
            tf.summary.trace_export(name='model_trace',step=0,profiler_outdir=sensorboard_file)
    
    
    # Save trained model, its parameters, and training history 
    save_trained_model(model, history, training_folder)
    return model, history, training_folder



'''
 Save trained model


'''
def save_trained_model(trained_model,history,training_folder,**kwargs):

    #-------- load hyperparameters------------# 
    hyperparams_file = training_folder + "/hyperparams.yaml"
    if os.path.isfile(hyperparams_file):
        fr = open(hyperparams_file, 'r')
        hyperparams = yaml.load(fr,Loader=yaml.BaseLoader)
        fr.close()
    else:
        print("Not Found hyper params file at {}".format(hyperparams_file))
        exit()


    #--------- save trained model and parameters----------#
    #i) save checkpoints
    checkpoint_folder = os.path.join(training_folder,'checkpoints')
    if(os.path.exists(checkpoint_folder)==False):
        os.makedirs(checkpoint_folder)
    checkpoint_name = 'my_checkpoint.ckpt'

    checkpoint = tf.train.Checkpoint(myAwesomeModel=trained_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory = checkpoint_folder,
                                                    checkpoint_name = checkpoint_name,
                                                    max_to_keep = 20)
    checkpoint_manager.save()
    
        
    #ii) save trained model
    saved_model_folder=os.path.join(training_folder,'trained_model')
    if(os.path.exists(saved_model_folder)==False):
        os.makedirs(saved_model_folder)
    saved_model_file = os.path.join(saved_model_folder,'my_model.h5')
    trained_model.save(saved_model_file)
    
    #iii) save training history
    # get the dictionary containing each metric and the loss for each epoch
    history_folder = os.path.join(training_folder,'train_process')
    history_file = os.path.join(history_folder,'my_history')

    # write it under the form of a json file
    with open(history_file,'w') as fd:
        json.dump(history.history, fd)


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




def train_model_narx(model, hyperparams, train_set, valid_set, training_mode='Integrative_way'):
    
    # crerate train results folder
    training_folder = pro_rd.create_training_files(hyperparams=hyperparams)
    
    for idx, data in enumerate(train_set):
        x = data[:,:int(hyperparams['features_num'])]
        y = data[:,int(hyperparams['features_num']):]
        y = y.reshape(-1)
        model.fit(x,y)

    x_valid = valid_set[0][:,:int(hyperparams['features_num'])]
    y_valid = valid_set[0][:,int(hyperparams['features_num']):]
    steps = DROPLANDING_PERIOD
    y_predict = model.forecast(x,y,steps,x_valid[:-1,:])
    pdb.set_trace()
    r2, rmse, mae, r_rmse = calculate_scores(y_valid[:-1,:],y_predict)

    return r2
    
    # Save trained model, its parameters, and training history 
    #save_trained_model(model, history, training_folder)
    #return model, history, training_folder

