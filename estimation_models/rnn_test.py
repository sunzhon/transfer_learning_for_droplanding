
from vicon_imu_data_process import process_landing_data as pro_rd
from estimation_models import rnn_models


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
    model_output, execution_time = rnn_models.model_forecast(trained_model, xy_test, hyperparams)
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

