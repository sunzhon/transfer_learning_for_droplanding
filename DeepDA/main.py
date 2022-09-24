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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import time as localtimepkg


current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../estimation_assessment")
sys.path.append(added_path)
import scores as es_sc
added_path = os.path.join(current_dir,"./../vicon_imu_data_process")
sys.path.append(added_path)
import process_landing_data as pro_rd
import const
from early_stopping import EarlyStopping



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
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--tst_domain', type=str, default=None)# Test dataset
    parser.add_argument('--labels_name',type=str, nargs='+')
    
    # training related
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
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
    parser.add_argument('--tgt_test_subjects_trials_len',type=dict,default={})

    # cross-validation
    parser.add_argument('--n_splits',type=int,default=0) # n_splits kfold or leaveone cross validation n_splits=0
    parser.add_argument('--early_stopping_patience',type=int, default=10) # patience for early stopping


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
    # src_domain, tgt_domain data to load
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)

    columns = args.features_name + args.labels_name

    source_subjects_trials_dataset = pro_rd.load_subjects_dataset(h5_file_name = folder_src, selected_data_fields=columns)
    target_subjects_trials_dataset = pro_rd.load_subjects_dataset(h5_file_name = folder_tgt, selected_data_fields=columns)

    # using the third dataset for testing: tst_domain
    if(args.tst_domain!=None):
        folder_tst = os.path.join(args.data_dir, args.tst_domain)
        test_subjects_trials_dataset = pro_rd.load_subjects_dataset(h5_file_name = folder_tst, selected_data_fields=columns)
        return [source_subjects_trials_dataset, target_subjects_trials_dataset, test_subjects_trials_dataset]

    return [source_subjects_trials_dataset, target_subjects_trials_dataset]


def load_data(args, src_subjects_trials_data, tgt_train_subjects_trials_data, tgt_test_subjects_trials_data):

    # data loader
    source_loader, n_labels = data_loader.load_motiondata(src_subjects_trials_data, args.batch_size, train=True, num_workers=args.num_workers, features_name=args.features_name, labels_name = args.labels_name)
    target_train_loader, _ = data_loader.load_motiondata(tgt_train_subjects_trials_data, args.batch_size, train=True, num_workers=args.num_workers,features_name=args.features_name, labels_name=args.labels_name)
    target_test_loader, _ = data_loader.load_motiondata(tgt_test_subjects_trials_data, 1, train=False, num_workers=args.num_workers, features_name=args.features_name,labels_name=args.labels_name)
    
    return source_loader, target_train_loader, target_test_loader, n_labels
    

def get_model(args):

    if(args.model_selection=='Normal_DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, target_reg_loss_weight=0).to(args.device)
    if(args.model_selection=='DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck,target_reg_loss_weight=1).to(args.device)
    if(args.model_selection=='Aug_DANN'):
        model = models.TransferNetForRegression(
                args.n_labels, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck,target_reg_loss_weight=1).to(args.device)
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

def test(model, target_test_loader, args):
    
    # model evaluation
    model.eval() # declare model evaluation, affects batch normalization and drop out layer
    test_loss = utils.AverageMeter()
    criterion = torch.nn.MSELoss()
    len_target_dataset = len(target_test_loader.dataset)
    test_acc = utils.AverageMeter()
    with torch.no_grad():
        for idx, (features, labels) in enumerate(target_test_loader):

            # load data to device
            features, labels = features.to(args.device), labels.to(args.device)
            predictions = model.predict(features)

            # loss
            loss = criterion(predictions, labels)
            test_loss.update(loss.item())

            # metrics: accuracy
            test_acc.update(es_sc.calculate_scores(labels.squeeze(dim=0).cpu().numpy(), predictions.squeeze(dim=0).cpu().numpy())[0])

            if(args.save_test):
                # transfer testing results' form into pandas Dataframe 
                pd_features = pd.DataFrame(data=torch.squeeze(features,dim=0).cpu().numpy(), columns=args.features_name)
                pd_labels = pd.DataFrame(data=torch.squeeze(labels,dim=0).cpu().numpy(), columns=args.labels_name)
                pd_predictions = pd.DataFrame(data=torch.squeeze(predictions,dim=0).cpu().numpy(), columns=args.labels_name)

                # create test results folder
                testing_folder = pro_rd.create_testing_files(args.training_folder)

                # save test results
                es_sc.save_test_result(pd_features, pd_labels, pd_predictions, testing_folder)
                
                # find the trail from which subject
                sum_trial_number=0
                for subject, trial_number in args.tgt_test_subjects_trials_len.items():
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

    return test_loss.avg, test_acc.avg, testing_folder


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader) # how many batchs
    len_target_loader = len(target_train_loader) # how many batchs, source and target should have similar batch numbers
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    print('source_loader len: {}, target_train_loader len:{}, n_batch: {}'.format(len_source_loader,len_target_loader, n_batch))

    # generate iterater of dataloader
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

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
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        for _ in range(n_batch):
            data_source, label_source = next(iter_source) # .next()
            data_target, label_target = next(iter_target) # .next()
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target, label_target = data_target.to(args.device), label_target.to(args.device)
            
            reg_loss, transfer_loss = model(data_source, data_target, label_source, label_target)
            '''

            if(transfer_loss.item()>0.1):
                loss = args.transfer_loss_weight * transfer_loss
            else:
                loss = args.regression_loss_weight*reg_loss + args.transfer_loss_weight * transfer_loss
            '''
            
            loss = args.regression_loss_weight*reg_loss + args.transfer_loss_weight * transfer_loss
            optimizer.zero_grad()
            '''
            print("\n===========开始迭代========")
            for name, params in model.named_parameters():
                print('--> name: ', name)
                print('--> param: ', params)
                print('--> grad_required: ', params.requires_grad)
                print('--> gard_value:', params.grad)
                print("===")
            '''
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            '''
            print("\n===========更新之后========")
            for name, params in model.named_parameters():
                print('--> name: ', name)
                print('--> param: ', params)
                print('--> grad_required: ', params.requires_grad)
                print('--> gard_value:', params.grad)
                print("===")
            print(optimizer)
            input("=======迭代结束========")
            '''

            train_loss_reg.update(reg_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            
        #log.append([train_loss_reg.avg, train_loss_transfer.avg, train_loss_total.avg])
        log.append([train_loss_reg.val, train_loss_transfer.val, train_loss_total.val])
        
        info = 'Epoch: [{:2d}/{}], reg_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        epoch, args.n_epoch, train_loss_reg.val, train_loss_transfer.val, train_loss_total.val)

        #ii) Test processing
        test_loss, test_acc, _ = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        # early stopping
        early_stop(test_loss, test_acc, model)
        if early_stop.early_stop:
            print(info)
            break
        print(info)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(os.path.join(training_folder,'best_model.pth')))
    args.save_test=True
    test_loss, test_acc, testing_folder = test(model, target_test_loader, args)
    print('Best result: {:.4f}'.format(early_stop.best_acc))
    # save trainied model
    torch.save(model.state_dict(),os.path.join(training_folder,'trained_model.pth'))

    return training_folder, testing_folder

def k_fold(args, multipe_domains_subjects_trials_data):

    if(args.n_splits>0):
        # k fold cross validation
        kf = KFold(n_splits=args.n_splits)
    else:
        # leave-one-out cross-validation
        kf = LeaveOneOut()

    # source and target subjects_trials
    src_subjects_trials_data = multipe_domains_subjects_trials_data[0]
    tgt_subjects_trials_data = multipe_domains_subjects_trials_data[1]
    tgt_subject_ids_names = list(tgt_subjects_trials_data.keys()) 
    print("target domain subjects: {}".format(tgt_subject_ids_names))

    for train_subject_indices, test_subject_indices in kf.split(tgt_subject_ids_names):
        #i) split target data into train and test target dataset 
        train_subject_ids_names = [tgt_subject_ids_names[subject_idx] for subject_idx in train_subject_indices]
        test_subject_ids_names = [tgt_subject_ids_names[subject_idx] for subject_idx in test_subject_indices]

        # specifiy test and train subjects for debug
        # i-1) choose data for training and testing
        if(args.model_selection in ['imu_augment', 'Aug_DANN', 'DANN', 'Normal_DANN',]):
            tst_subjects_trials_data = multipe_domains_subjects_trials_data[2] # additional domain for only test, it is raw target domain
            tgt_train_subjects_trials_data = {subject_id_name: tgt_subjects_trials_data[subject_id_name] for subject_id_name in train_subject_ids_names}
            tgt_test_subjects_trials_data = {subject_id_name: tst_subjects_trials_data[subject_id_name] for subject_id_name in test_subject_ids_names}
        else: #baseline, pretrain, finetuning, and DANN with repeated IMU
            tgt_train_subjects_trials_data = {subject_id_name: tgt_subjects_trials_data[subject_id_name] for subject_id_name in train_subject_ids_names}
            tgt_test_subjects_trials_data = {subject_id_name: tgt_subjects_trials_data[subject_id_name] for subject_id_name in test_subject_ids_names}

        args.test_subject_ids_names = test_subject_ids_names
        print('test subjects id names: {}'.format(test_subject_ids_names))

        #ii) load dataloader accodring to source and target subjects_trials_dataset
        source_loader, target_train_loader, target_test_loader, n_labels = load_data(args, src_subjects_trials_data, tgt_train_subjects_trials_data, tgt_test_subjects_trials_data)
        args.n_labels = n_labels
        if args.epoch_based_training:
            args.max_iter = args.n_epoch * min(len(source_loader), len(target_train_loader))
            print(' epoch based tranining')
        else:
            args.max_iter =   args.n_epoch * args.n_iter_per_epoch
            print(' epoch based tranining')

        # test subjects and trials, {subject_id_name:len(['01','02',...]), ...}
        tgt_test_subjects_trials_len = {subject_id_name: len(list(trials.keys())) for subject_id_name, trials in tgt_test_subjects_trials_data.items()} 
        args.tgt_test_subjects_trials_len=tgt_test_subjects_trials_len


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

        #vi) train model
        training_folder, testing_folder = train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)
    
    
def main():
    #1) hyper parameters
    parser = get_parser()
    parser.set_defaults(backbone='mlnn') # backbone model selection
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print("device:", args.device)

    #data path
    args.data_dir = const.DATA_PATH


    
    # features
    features_name = ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST, const.ACC_GYRO_FIELDS)
    setattr(args, "features_name", features_name)
    setattr(args, "n_labels", 1)


    # train , this max_iter iss important for DANN train
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)

    set_random_seed(args.seed)

    #2) open datafile and load data
    #src_subjects_trials_data, tgt_subjects_trials_data = open_datafile(args)
    multipe_domains_subjects_trials_data = open_datafile(args)

    #3) cross_validation for training and evaluation model
    setattr(args, 'test_subject_ids_names', [])

    #k_fold(args,src_subjects_trials_data, tgt_subjects_trials_data)
    k_fold(args, multipe_domains_subjects_trials_data)

    



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
