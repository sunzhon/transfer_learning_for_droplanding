#! /bin/pyenv python
#coding: --utf-8
import configargparse
import data_loader
import os
import sys
import torch
import torchmetrics
import models
import utils
from utils import str2bool
import numpy as np
import random
import pdb
import pandas as pd
import yaml
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg
import itertools
import pickle
from thop import profile


current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../assessments")
sys.path.append(added_path)
import scores as es_sc
added_path = os.path.join(current_dir,"./../vicon_imu_data_process")
sys.path.append(added_path)
import process_landing_data as pro_rd
import const
from early_stopping import EarlyStopping

# release gpu memory
torch.cuda.empty_cache()


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # description of this config or code modification
    parser.add("--config_name", type=str, default="dann_1")
    parser.add("--config_comments", type=str, default="dann")
    parser.add("--result_folder", type=str, default=None)

    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=False)

    # data loading related
    parser.add_argument('--dataset_name', type=str, default='original')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tcl_domain', type=str, required=True)
    parser.add_argument('--tre_domain', type=str, required=True)
    parser.add_argument('--tst_domain', type=str, required=True)# Test dataset
    parser.add_argument('--scaler_file', type=str, required=True) # patience for early stopping
    parser.add_argument('--labels_name',type=str, nargs='+')
    parser.add_argument('--features_name',type=str, nargs='+')
    parser.add_argument('--landing_manner',type=str, default='double_legs')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=True, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--regression_loss_weight', type=float, default=1.0)
    parser.add_argument('--transfer_loss_weight', type=float, default=1.0)
    parser.add_argument('--transfer_loss', type=str, default='adv')

    parser.add_argument('--save_test', type=bool, default=False)
    parser.add_argument('--training_folder', type=str, default=None)

    # model selections
    parser.add_argument('--model_name', type=str, default='DANN')
    parser.add_argument('--trained_model_state_path', type=str, default='None')

    # a trial from a subject in a test
    parser.add_argument('--test_subjects',type=str,default='None')
    parser.add_argument('--trial_idx',type=int,default=0)
    parser.add_argument('--tst_test_subjects_trials_len',type=dict,default={})
    parser.add_argument('--tre_train_subjects_trials_len',type=dict,default={})

    # cross-validation
    parser.add_argument('--n_splits',type=int,default=0) # n_splits kfold or leaveone cross validation n_splits=0
    parser.add_argument('--early_stopping_patience',type=int, default=10) # patience for early stopping
    parser.add_argument('--use_early_stop',type=str2bool, default=True) # patience for early stopping

    # sub_num subjects and tst_trial_num and tre_trial_num trials of the subjects will be used
    parser.add_argument('--sub_num',type=int, default=17) # patience for early stopping
    parser.add_argument('--tst_trial_num',type=int, default=10) # patience for early stopping
    parser.add_argument('--tre_trial_num',type=int, default=25) # patience for early stopping

    # train_sub_num subjects were used to train model 
    parser.add_argument('--train_sub_num',type=int, default=14) # patience for early stopping
    parser.add_argument('--test_sub_num',type=int, default=1) # patience for early stopping

    # layer num of features
    parser.add_argument('--feature_layer_num',type=int, default=1) # patience for early stopping

    # the cross-validation num. How many combination of (train sub and test subs) are used 
    parser.add_argument('--cv_num',type=int, default=10) # patience for early stopping

    # time complexity and model complexity
    parser.add_argument('--FLOPs',type=float, default=.0) # computational cost of the model
    parser.add_argument('--Params',type=float, default=.0) # model complexity of the model


    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def open_datafile(args):
    '''
    Load dataset for tst and tre

    '''
    # dataset fields
    columns = args.features_name + args.labels_name

    # src_domain, tgt_domain data to load
    domains = [args.src_domain, args.tcl_domain, args.tre_domain, args.tst_domain]
    domains_name = ['src', 'tcl', 'tre', 'tst']
    multiple_domain_datasets= {}
    for domain, domain_name in zip(domains, domains_name):
        if domain !='None':
            domain_data_folder = os.path.join(args.data_dir, domain) # source data
            domain_dataset, dataset_columns = pro_rd.load_subjects_dataset(data_file_name=domain_data_folder, selected_data_fields=columns)
            multiple_domain_datasets[domain_name] = domain_dataset
    
    return multiple_domain_datasets, dataset_columns



def get_dataloader(args, multiple_domain_datasets):
    '''
    get dataloader and put the datasets {subject: {trial:,[], trial_1: []}}...
    the element of a dataloader is a table represent a batch of trial. ([batch_size, seq_len, features], [batch_size, seq_len, label].
    where, a trial of data is  (seq, [features, labels])

    '''

    # data loader
    domain_data_loaders={}
    for domain_name, domain_data in multiple_domain_datasets.items():
        if(domain_name!='tst'):
            domain_data_loaders[domain_name], n_labels = data_loader.load_motiondata(domain_data, args.batch_size, train=True, num_workers=args.num_workers, features_name=args.features_name, labels_name = args.labels_name)
        else:
            domain_data_loaders[domain_name], n_labels = data_loader.load_motiondata(domain_data,1, train=False, num_workers=args.num_workers, features_name=args.features_name, labels_name = args.labels_name)

    return domain_data_loaders, n_labels


def get_model(args):
    num_layers = args.feature_layer_num
    if(args.model_name=='Normal_DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=0).to(args.device)

    elif(args.model_name=='DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=1).to(args.device)

    elif(args.model_name=='Aug_DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=1).to(args.device)

    elif(args.model_name=='baseline'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', features_num = len(args.features_name),num_layers=num_layers).to(args.device)

    elif(args.model_name=='augmentation'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', features_num = len(args.features_name),  num_layers=num_layers).to(args.device)

    elif(args.model_name=='baseline_cnn'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='cnn', num_layers=num_layers).to(args.device)

    elif(args.model_name=='pretrained'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', num_layers=num_layers).to(args.device)

    elif(args.model_name=='finetuning'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', num_layers = num_layers, finetuning=True).to(args.device)
        model.load_state_dict(torch.load(os.path.join(const.RESULTS_PATH, args.trained_model_state_path, 'trained_model.pth')))

    elif(args.model_name=='discriminator'):
        model = models.DiscriminatorModel(num_label=args.n_labels, base_net='discriminator').to(args.device)

    else:
        exit("MODEL is not exist")

    print('Model selection: {}'.format(args.model_name))

    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    #optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, test_data_loader, args, **kwargs):
    
    # model evaluation
    model.eval() # declare model evaluation, affects batch normalization and drop out layer
    test_loss = utils.AverageMeter()
    criterion = torch.nn.MSELoss()
    len_target_dataset = len(test_data_loader.dataset)
    test_acc = utils.AverageMeter()
    r2score = torchmetrics.R2Score(2).to(args.device)#num_outputs=2,multioutput='raw_values'
    with torch.no_grad(): # no do calcualte grad
        for trial_idx, (features, labels) in enumerate(test_data_loader):

            # load data to device
            features, labels = features.to(args.device), labels.to(args.device)
            predictions = model.predict(features)

            # loss
            loss = criterion(predictions, labels)
            test_loss.update(loss.item())
            
            predictions = predictions.squeeze(0)
            labels = labels.squeeze(0)

            # metrics: accuracy
            mean_r2 = r2score(predictions, labels)
            test_acc.update(mean_r2)

            if(args.save_test):
                # transfer tensors to numpy array
                a_label = labels.cpu().numpy()
                a_prediction = predictions.cpu().numpy()
                features = features.squeeze(0).cpu().numpy()

                # load scaler, only for testing 
                scaler = pickle.load(open(os.path.join(const.DATA_PATH, args.scaler_file),'rb'))

                # inverse transform data
                feature_label_index = [args.dataset_columns.index(fl) for fl in (args.features_name + args.labels_name)]

                tmp_dataset = np.zeros((features.shape[0],len(args.dataset_columns)))

                tmp_dataset[:,feature_label_index] = np.concatenate([features,a_label],axis=1)
                unscaled_feature_labels = scaler.inverse_transform(tmp_dataset)[:,feature_label_index]

                tmp_dataset[:,feature_label_index] = np.concatenate([features,a_prediction],axis=1)
                unscaled_feature_predictions = scaler.inverse_transform(tmp_dataset)[:,feature_label_index]

                unscaled_features = unscaled_feature_labels[:,:-a_label.shape[1]]
                unscaled_labels = unscaled_feature_labels[:,-a_label.shape[1]:]
                unscaled_predictions = unscaled_feature_predictions[:,-a_label.shape[1]:]

                # transfer testing results' form into pandas Dataframe 
                pd_features = pd.DataFrame(data=unscaled_features, columns=args.features_name)
                pd_labels = pd.DataFrame(data=unscaled_labels, columns=args.labels_name)
                pd_predictions = pd.DataFrame(data=unscaled_predictions, columns=args.labels_name)

                # create test results folder
                testing_folder = pro_rd.create_testing_files(args.training_folder)

                # save test results
                es_sc.save_test_result(pd_features, pd_labels, pd_predictions, testing_folder)
                # trial idx
                args.trial_idx = trial_idx
                
                # find the trail from which subject, stop the test early
                # save hyper parameters
                hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
                with open(hyperparams_file,'w') as fd:
                    yaml.dump(vars(args), fd)
            else:
                testing_folder = None

            print("test trial:{}".format(trial_idx))
            if('test_times' in kwargs.keys()): # just for checing the acc of train dataset during train progress
                if(trial_idx > kwargs['test_times']): # just test few times, e.g., 4
                    break

    return test_loss.avg, test_acc.avg, testing_folder


def train(domain_data_loaders,  model, optimizer, lr_scheduler, n_batch, args):

    # log information
    log = []

    # crerate train results folder
    training_folder = pro_rd.create_training_files(hyperparams=vars(args))
    args.save_test = False 
    args.training_folder = training_folder

    # initialize the early_stopping object
    early_stop = EarlyStopping(save_path=training_folder, patience=args.early_stopping_patience, verbose=True)
    
    # conduct all epochs
    for epoch in range(1, args.n_epoch+1):
        # i) Train processing
        model.train()
        train_loss_reg = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()

        # generate iterater of dataloader
        domains_data_iter={}
        for name, a_data_loader in domain_data_loaders.items(): # name and dataset of a domain dataloader
            if name!='tst': # do not need to iter tst data
                domains_data_iter[name] = iter(a_data_loader)
        
        # conduct all batchs 
        for _ in range(n_batch): # batch
            domains_samples = {}
            for name in domains_data_iter.keys(): # domains
                data, label = next(domains_data_iter[name])
                data, label = data.to(args.device), label.to(args.device)
                domains_samples[name] = (data,label) # pack batched data and label of each domain
            reg_loss, transfer_loss = model(domains_samples)
            reg_loss = args.regression_loss_weight*reg_loss 
            transfer_loss = args.transfer_loss_weight * transfer_loss
            
            loss = reg_loss + transfer_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_reg.update(reg_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        #log.append([train_loss_reg.avg, train_loss_transfer.avg, train_loss_total.avg])
        log.append([train_loss_reg.val, train_loss_transfer.val, train_loss_total.val])
        
        info = 'Epoch: [{:2d}/{}], reg_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}\n'.format(
                        epoch, args.n_epoch, train_loss_reg.val, train_loss_transfer.val, train_loss_total.val)

        # Train processing Acc
        #ii) Test processing Acc,
        test_loss, test_acc, _ = test(model, domain_data_loaders['tst'], args)
        info += 'test_loss {:.4f}, test_acc: {:.4f}\n'.format(test_loss, test_acc)

        #iii) log save
        np_log = np.array(log, dtype=float)
        train_log_path = os.path.join(training_folder,'train_log.csv')
        np.savetxt(train_log_path, np_log, delimiter=',', fmt='%.6f')

        # early stopping
        early_stop(test_loss, test_acc, model)
        if early_stop.early_stop and args.use_early_stop:
            print(info)
            break
        print(info)

    # load the last checkpoint with the best model for test model 
    model.load_state_dict(torch.load(os.path.join(training_folder,'best_model.pth')))
    args.save_test = True
    test_loss, test_acc, testing_folder = test(model, domain_data_loaders['tst'], args)
    print('BEST RESULT: r2: {:.4f}, loss: {:.4f}'.format(early_stop.best_acc,early_stop.best_score))
    print('Its training folder: {}\n\n'.format(training_folder))
    # save trainied model
    torch.save(model.state_dict(),os.path.join(training_folder,'trained_model.pth'))

    return training_folder, testing_folder

def k_fold(args, multiple_domain_datasets):

    if(args.n_splits>0):
        # k fold cross validation
        kf = KFold(n_splits=args.n_splits)
    else:
        # leave-one-out cross-validation
        kf = LeaveOneOut()

    # copy dataset
    load_domain_dataset = {}# copy.deepcopy(multiple_domain_datasets)

    # NOTE: tre and tst mush have same subjects
    assert(set(multiple_domain_datasets['tre'])==set(multiple_domain_datasets['tst'])) # reg target and test target have same subject list, which using labels

    # print loaded data domain name
    for name, data in multiple_domain_datasets.items():
        print("{} domain subjects: {}\n".format(name, list(data.keys())))

    tst_subject_ids_names = list(multiple_domain_datasets['tst'].keys()) 

    '''
    for train_subject_indices, test_subject_indices in kf.split(tst_subject_ids_names): # typical CV
    '''
    train_test_list = list(range(len(tst_subject_ids_names))) # all subjects [0,1,2,3,....]
    train_sub_num = args.train_sub_num # select how many subjects for training
    test_sub_num = args.test_sub_num # select howm many subjects for testing
    train_indices_list = list(itertools.combinations(train_test_list,train_sub_num)) # train_sub_num subjects for training, remaining subjects for test -CV
    random_selected_train_index = random.sample(train_indices_list, args.cv_num if len(train_indices_list) > args.cv_num else len(train_indices_list)) # 随机选出cv_num (e.g., 15) 种训练对象的组合, selected cv_num combiantions
    for loop, train_subject_indices in enumerate(random_selected_train_index): # leave-CV
        non_train_subject_list =  list(set(train_test_list)-set(train_subject_indices))# not used in training dataset
        test_indices_list = list(itertools.combinations(non_train_subject_list, test_sub_num if len(non_train_subject_list) > test_sub_num else len(non_train_subject_list))) # list of test indices
        test_subject_indices = random.sample(test_indices_list, 1)[0] # output is [(,),(,),...], random select one list, [0] means the one.

        #i) split target data into train and test target dataset 
        train_subject_ids_names = [tst_subject_ids_names[subject_idx] for subject_idx in train_subject_indices]
        test_subject_ids_names = [tst_subject_ids_names[subject_idx] for subject_idx in test_subject_indices]
        args.test_subjects = test_subject_ids_names
        print("A kfold loop....")
        print('train subjects id names: {}'.format(train_subject_ids_names))
        print('test subjects id names: {}\n'.format(test_subject_ids_names))

        # specifiy test and train subjects
        #i-1) choose data for regssion training (using label) and testing base on train and test subject indices
        tre_train_dataset = {subject_id_name: {key: value for idx, (key, value) in enumerate(multiple_domain_datasets['tre'][subject_id_name].items()) if idx < args.tre_trial_num} for subject_id_name in train_subject_ids_names}
        tst_test_dataset = {subject_id_name: {key: value for idx, (key, value) in enumerate(multiple_domain_datasets['tst'][subject_id_name].items()) if idx < args.tst_trial_num} for subject_id_name in test_subject_ids_names}

        load_domain_dataset['tre'] = tre_train_dataset
        load_domain_dataset['tst'] = tst_test_dataset

        # the length of test subjects and trials, {subject_id_name:len(['01','02',...]), ...}
        tre_train_subjects_trials_len = {subject_id_name: len(trials.keys()) for subject_id_name, trials in tre_train_dataset.items()} 
        tst_test_subjects_trials_len = {subject_id_name: len(trials.keys()) for subject_id_name, trials in tst_test_dataset.items()} 

        args.tre_train_subjects_trials_len = tre_train_subjects_trials_len
        args.tst_test_subjects_trials_len = tst_test_subjects_trials_len

        print('trial number of tre train dataset: {}'.format(args.tre_train_subjects_trials_len))
        print('trial number of tst test dataset: {}\n'.format(args.tst_test_subjects_trials_len))

        #ii) load dataloader accodring to source and target subjects_trials_dataset
        domain_data_loaders, n_labels = get_dataloader(args, load_domain_dataset)
        args.n_labels = n_labels

        #iii) load model
        set_random_seed(args.seed)
        model = get_model(args)

        # get model time complexity and memory complexisty
        args.FLOPs, args.Params = np.array(profile(model.base_network, inputs=(torch.randn(1,model.base_network.seq_len,len(args.features_name)).to(args.device),))) + np.array(profile(model.output_layer, inputs=(torch.randn(100,200).to(args.device),)))

        #iv) get optimizer
        optimizer = get_optimizer(model, args)
        
        #v) learner scheduler
        if args.lr_scheduler:
            scheduler = get_scheduler(optimizer, args)
        else:
            scheduler = None

        loader_len=[]
        for name, data in domain_data_loaders.items():
            loader_len.append(len(data))
            print("{} data's len: {}".format(name,len(data)))
        n_batch = min(loader_len)
        if n_batch == 0:
            n_batch = args.n_iter_per_epoch 
        print("n_batch: {}".format(n_batch))

        if args.epoch_based_training:
            args.max_iter = args.n_epoch * n_batch
            print('Epoch based tranining')
        else:
            args.max_iter =   args.n_epoch * args.n_iter_per_epoch
            print('Interation based tranining')

        #vi) train model
        training_folder, testing_folder = train(domain_data_loaders, model, optimizer, scheduler, n_batch, args)
        
def model_evaluation(args, multiple_domain_datasets):

    # load model
    model = models.BaselineModel(num_label=args.n_labels, base_net=args.backbone).to(args.device)
    model.load_state_dict(torch.load(os.path.join(const.RESULTS_PATH, args.trained_model_state_path, 'trained_model.pth')))

    # load data
    tst_subjects_trials_data = multiple_domain_datasets['tst'] # additional domain for only test, it is raw target domain
    tst_subject_ids_names = list(tst_subjects_trials_data.keys()) 
    print("test domain subjects: {}".format(tst_subject_ids_names))

    for subject in tst_subject_ids_names:
        test_subjects_trials_data = {subject: tst_subjects_trials_data[subject]}
        target_test_loader, _ = data_loader.load_motiondata(test_subjects_trials_data, 1, train=False, num_workers=args.num_workers, features_name=args.features_name,labels_name=args.labels_name)

        # load the last checkpoint with the best model
        args.save_test=True
        args.training_folder = os.path.join(os.path.dirname(os.path.join(const.RESULTS_PATH, args.trained_model_state_path)),'model_evaluation')
        test_loss, test_acc, testing_folder = test(model, target_test_loader, args)
        print('Test subject: {}, Test loss{:.4f}, test acc: {:.4f}'.format(subject, test_loss, test_acc))
        print('Test results save at :{}'.format(testing_folder))
    
def main():
    ##torch.multiprocessing.set_start_method('spawn') # set this, need to set work_num=0
    #1) hyper parameters
    parser = get_parser()
    parser.set_defaults(backbone='mlnn') # backbone model selection
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print("device:", args.device)

    #data path
    args.data_dir = const.DATA_PATH
    
    # features
    setattr(args, "n_labels", 1)

    # train , this max_iter iss important for DANN train
    # for SGD
    args.lr_gamma =1/args.n_epoch*10

    set_random_seed(args.seed)

    #2) open datafile and load data
    multiple_domain_datasets, dataset_columns = open_datafile(args)
    setattr(args, "dataset_columns", list(dataset_columns))

    #3) cross_validation for training and evaluation model
    setattr(args, 'test_subjects', [])

    # test or cross validation training
    if(args.model_name=='test_model'):
        model_evaluation(args, multiple_domain_datasets) # args contains model_param path
    else:
        k_fold(args, multiple_domain_datasets)

    


if __name__ == "__main__":
    main()


'''
    # compare models' parameters
    model_1 = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', finetuning=True).to(args.device)
    model_2 = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', finetuning=True).to(args.device)
    model_1.load_state_dict(torch.load(os.path.join(const.RESULTS_PATH, 'training_testing/pretrain_finetuning/training_073632', 'trained_model.pth'),map_location=torch.device('cpu')))
    model_2.load_state_dict(torch.load(os.path.join(const.RESULTS_PATH, 'training_testing/pretrain_finetuning/training_074653', 'trained_model.pth'),map_location=torch.device('cpu')))
    for param1, param2 in zip(model_1.parameters(), model_2.parameters()):
        print('model1 param name:{}, model2 param name:{}'.format(param1.name, param2.name))
        pdb.set_trace()
        if(param1.data!=param2.data):
            print('model1 name: {} and data:{}:'.format(param1.name, param1.data))
            print('model2 name: {} and data:{}:'.format(param2.name, param2.data))

'''
