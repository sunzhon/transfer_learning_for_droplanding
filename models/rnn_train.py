
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
def save_trained_model(trained_model, history=None, training_folder=None, **kwargs):

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
    if(history!=None):
        # get the dictionary containing each metric and the loss for each epoch
        history_folder = os.path.join(training_folder,'train_process')
        history_file = os.path.join(history_folder,'my_history')

        # write it under the form of a json file
        with open(history_file,'w') as fd:
            json.dump(history.history, fd)





'''

def model_narx(hyperparams):
    exog_order = hyperparams['features_num'] * [5]
    auto_order = 5
    mdl = NARX(LinearRegression(), auto_order = auto_order, exog_order = exog_order)
    return mdl
'''


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

