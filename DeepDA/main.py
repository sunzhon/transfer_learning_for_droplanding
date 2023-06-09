#! /bin/pyenv python
#coding: --utf-8
import configargparse
import data_loader
import os
import sys
import torch
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


current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../estimation_assessment")
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
    parser.add("--config_alias_name", type=str, default="dann_1")
    parser.add("--config_comments", type=str, default="dann")
    parser.add("--investigation_results_folder", type=str, default=None)

    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=False)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tcl_domain', type=str, required=True)
    parser.add_argument('--tre_domain', type=str, required=True)
    parser.add_argument('--tst_domain', type=str, required=True)# Test dataset
    parser.add_argument('--labels_name',type=str, nargs='+')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=40)
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
    parser.add_argument('--model_selection', type=str, default='DANN')
    parser.add_argument('--trained_model_state_path', type=str, default='None')

    # a trial from a subject in a test
    parser.add_argument('--test_subject',type=str,default='None')
    parser.add_argument('--tst_test_subjects_trials_len',type=dict,default={})

    # cross-validation
    parser.add_argument('--n_splits',type=int,default=0) # n_splits kfold or leaveone cross validation n_splits=0
    parser.add_argument('--early_stopping_patience',type=int, default=10) # patience for early stopping
    parser.add_argument('--use_early_stop',type=str2bool, default=True) # patience for early stopping

    # dataset loading mode, if this is not None, than use this mode, the data domain created in main.py
    parser.add_argument('--train_sub_num',type=int, default=5) # patience for early stopping
    parser.add_argument('--train_trial_num',type=int, default=5) # patience for early stopping


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

    # dataset fields
    columns = args.features_name + args.labels_name

    # src_domain, tgt_domain data to load
    domains = [args.src_domain, args.tcl_domain, args.tre_domain, args.tst_domain]
    domains_name = ['src', 'tcl', 'tre', 'tst']
    multiple_domain_datasets= {}
    for domain, domain_name in zip(domains, domains_name):
        if domain !='None':
            domain_data_folder = os.path.join(args.data_dir, domain) # source data
            domain_dataset = pro_rd.load_subjects_dataset(h5_file_name = domain_data_folder, selected_data_fields=columns)
            multiple_domain_datasets[domain_name] = domain_dataset
    
    '''
     multiple_domain_datasets= {}
     domains_name = ['src', 'tcl', 'tre', 'tst']
     main_data_folder = os.path.join(args.data_dir, args.online_select_dataset) # source data
     all_dataset = pro_rd.load_subjects_dataset(h5_file_name = main_data_folder, selected_data_fields=columns)

     multiple_domain_datasets['src'] = copy.deepcopy(all_dataset) #source domain, not used
     multiple_domain_datasets['tcl'] = copy.deepcopy(all_dataset) # target classfication, not used
     multiple_domain_datasets['tre'] = copy.deepcopy(all_dataset) # target regression, used for basline
     multiple_domain_datasets['tst'] = copy.deepcopy(all_dataset) # target test, used for basline
    '''

    return multiple_domain_datasets



def load_data(args, multiple_domain_datasets):

    # data loader
    domain_data_loaders={}
    for domain_name, domain_data in multiple_domain_datasets.items():
        if(domain_name!='tst'):
            a_data_loader, n_labels = data_loader.load_motiondata(domain_data, args.batch_size, train=True, num_workers=args.num_workers, features_name=args.features_name, labels_name = args.labels_name)
        else:
            a_data_loader, n_labels = data_loader.load_motiondata(domain_data,1, train=False, num_workers=args.num_workers, features_name=args.features_name, labels_name = args.labels_name)

        domain_data_loaders[domain_name] = a_data_loader

    return domain_data_loaders, n_labels


def get_model(args):
    if(args.model_selection=='Normal_DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=0).to(args.device)

    if(args.model_selection=='DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=1).to(args.device)

    if(args.model_selection=='Aug_DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=1).to(args.device)

    if(args.model_selection=='baseline'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn').to(args.device)

    if(args.model_selection=='imu_augment'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn').to(args.device)

    if(args.model_selection=='pretrained'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn').to(args.device)

    if(args.model_selection=='finetuning'):
        model = models.BaselineModel(num_label=args.n_labels, base_net='mlnn', finetuning=True).to(args.device)
        model.load_state_dict(torch.load(os.path.join(const.RESULTS_PATH, args.trained_model_state_path, 'trained_model.pth')))

    if(args.model_selection=='discriminator'):
        model = models.DiscriminatorModel(num_label=args.n_labels, base_net='discriminator').to(args.device)

    print('Model selection: {}'.format(args.model_selection))

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
    with torch.no_grad():
        for idx, (features, labels) in enumerate(test_data_loader):

            # load data to device
            features, labels = features.to(args.device), labels.to(args.device)
            predictions = model.predict(features)

            # loss
            loss = criterion(predictions, labels)
            test_loss.update(loss.item())
            
            # metrics: accuracy
            '''
            r2=[]
            for idx in range(labels.shape[0]):
                a_label = labels[idx,:,:].cpu().numpy()
                a_prediction = predictions[idx,:,:].cpu().numpy()
                r2.append(es_sc.calculate_scores(a_label, a_prediction)[0])
            mean_r2 =  sum(r2)/len(r2)
            '''

            mean_r2 = torch.mean(f1_score(labels, predictions))
            test_acc.update(mean_r2)

            if(args.save_test):
                # transfer testing results' form into pandas Dataframe 
                a_label = labels[0,:,:].cpu().numpy()
                a_prediction = predictions[0,:,:].cpu().numpy()

                pd_features = pd.DataFrame(data=torch.squeeze(features,dim=0).cpu().numpy(), columns=args.features_name)
                pd_labels = pd.DataFrame(data=a_label, columns=args.labels_name)
                pd_predictions = pd.DataFrame(a_prediction, columns=args.labels_name)

                # create test results folder
                testing_folder = pro_rd.create_testing_files(args.training_folder)

                # save test results
                es_sc.save_test_result(pd_features, pd_labels, pd_predictions, testing_folder)
                
                # find the trail from which subject
                sum_trial_number=0
                for subject, trial_number in args.tst_test_subjects_trials_len.items():
                    #print("idx: {} and trial number: {}".format(idx, trial_number))
                    sum_trial_number+=trial_number
                    if(idx < sum_trial_number):
                        args.test_subject = subject
                        #print('test_subject',subject)
                        break;

                # save hyper parameters
                hyperparams_file = os.path.join(testing_folder,"hyperparams.yaml")
                with open(hyperparams_file,'w') as fd:
                    yaml.dump(vars(args), fd)
            else:
                testing_folder = None

            if('train_acc_check_times' in kwargs.keys()): # just for checing the acc of train dataset during train progress
                if(idx>kwargs['train_acc_check_times']): # just test few times, e.g., 4
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

    for epoch in range(1, args.n_epoch+1):
        # i) Train processing
        model.train()
        train_loss_reg = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()

        # generate iterater of dataloader
        domains_data_iter={}
        for name, a_data_loader in domain_data_loaders.items():
            if name!='tst': # do not need to iter tst data
                domains_data_iter[name] = iter(a_data_loader)
        
        for _ in range(n_batch):
            domains_samples = {}
            for name in domains_data_iter.keys():
                data, label = next(domains_data_iter[name])
                data, label = data.to(args.device), label.to(args.device)
                domains_samples[name] = (data,label)

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
        #_, train_src_acc, _ =test(model, source_loader, args, train_acc_check_times=1)
        #_, train_tgt_acc, _ =test(model, target_reg_train_loader, args, train_acc_check_times=1)
        #info += 'train_src_acc {:.4f}, train_tgt_acc: {:.4f}\n'.format(train_src_acc, train_tgt_acc)

        #ii) Test processing Acc,
        test_loss, test_acc, _ = test(model, domain_data_loaders['tst'], args)
        info += 'test_loss {:.4f}, test_acc: {:.4f}\n'.format(test_loss, test_acc)

        #iii) log save
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')

        # early stopping
        early_stop(test_loss, test_acc, model)
        if early_stop.early_stop and args.use_early_stop:
            print(info)
            break
        print(info)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(os.path.join(training_folder,'best_model.pth')))
    args.save_test=True
    test_loss, test_acc, testing_folder = test(model, domain_data_loaders['tst'], args)
    print('Best result: {:.4f}'.format(early_stop.best_acc))
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
    load_domain_dataset = copy.deepcopy(multiple_domain_datasets)

    # NOTE: tre and tst mush have same subjects
    assert(set(multiple_domain_datasets['tre'])==set(multiple_domain_datasets['tst'])) # reg target and test target have same subject list, which using labels
    # print loaded data domain name
    for name, data in multiple_domain_datasets.items():
        print("{} domain subjects: {}\n".format(name, list(data.keys())))

    tst_subject_ids_names = list(multiple_domain_datasets['tst'].keys()) 

    '''
    for train_subject_indices, test_subject_indices in kf.split(tst_subject_ids_names): # typical CV
    '''
    train_test_list = list(range(len(tst_subject_ids_names))) # all subjects
    train_sub_num = args.train_sub_num # select how many subjects for training, and remaining for test
    train_index = list(itertools.combinations(train_test_list,train_sub_num)) # train_sub_num subjects for training, remaining subjects for test -CV
    random_selected_train_index = random.sample(train_index, 15) # 随机选出15种训练对象的组合, selected 15 combiantions
    for loop, train_subject_indices in enumerate(random_selected_train_index): # leave-CV
        test_subject_indices = list(set(train_test_list)-set(train_subject_indices)) # leave-CV
        if(len(test_subject_indices)>3):
            test_subject_indices = random.sample(test_subject_indices,3) # random select 5 subjects from the all test subjects


        #i) split target data into train and test target dataset 
        train_subject_ids_names = [tst_subject_ids_names[subject_idx] for subject_idx in train_subject_indices]
        test_subject_ids_names = [tst_subject_ids_names[subject_idx] for subject_idx in test_subject_indices]
        args.test_subject_ids_names = test_subject_ids_names
        print('train subjects id names: {}'.format(train_subject_ids_names))
        print('test subjects id names: {}\n'.format(test_subject_ids_names))

        # specifiy test and train subjects
        #i-1) choose data for regssion training (using label) and testing
        tre_train_dataset = {subject_id_name: multiple_domain_datasets['tre'][subject_id_name] for subject_id_name in train_subject_ids_names}
        tst_test_dataset = {subject_id_name: multiple_domain_datasets['tst'][subject_id_name] for subject_id_name in test_subject_ids_names}
        load_domain_dataset['tre'] = tre_train_dataset
        load_domain_dataset['tst'] = tst_test_dataset

        # test subjects and trials, {subject_id_name:len(['01','02',...]), ...}
        tst_test_subjects_trials_len = {subject_id_name: len(list(trials.keys())) for subject_id_name, trials in tst_test_dataset.items()} 
        args.tst_test_subjects_trials_len=tst_test_subjects_trials_len

        #ii) load dataloader accodring to source and target subjects_trials_dataset
        domain_data_loaders, n_labels = load_data(args, load_domain_dataset)
        args.n_labels = n_labels

        #iii) load model
        set_random_seed(args.seed)
        model = get_model(args)

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

    #load data
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
    features_name = ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST, const.ACC_GYRO_FIELDS)
    setattr(args, "features_name", features_name)
    setattr(args, "n_labels", 1)


    # train , this max_iter iss important for DANN train
    # for SGD
    args.lr_gamma =1/args.n_epoch*10

    set_random_seed(args.seed)

    #2) open datafile and load data
    multiple_domain_datasets = open_datafile(args)

    #3) cross_validation for training and evaluation model
    setattr(args, 'test_subject_ids_names', [])

    #k_fold(args,src_subjects_trials_data, tgt_subjects_trials_data)
    if(args.model_selection=='test_model'):
        model_evaluation(args, multiple_domain_datasets)
    else:
        k_fold(args, multiple_domain_datasets)

    


def f1_score(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    '''
    assert(y_true.shape==y_pred.shape)
    r2 = torch.mean(1-torch.sum((y_true-y_pred)**2,(1,2))/torch.sum((y_true-torch.mean(y_true,1).unsqueeze(2))**2,(1,2)))
    
    return r2



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
