from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
added_path = os.path.join(current_dir,"./../vicon_imu_data_process")
sys.path.append(added_path)
import process_landing_data as pro_rd
import const


class MotionDataset(Dataset):
    def __init__(self, data_file, features_name, labels_name, transform=None,label_transform=None, device='cuda'):
        if(isinstance(data_file,str)):
            subjects_trials_dataset, _ = pro_rd.load_subjects_dataset(data_file_name = data_file, selected_data_fields=features_name+labels_name)
        elif(isinstance(data_file,dict)): # using this option.
            subjects_trials_dataset = data_file
        else:
            print('data_file is wrong: ', data_file)
            exit()

        self.features = []
        self.labels = []
        self.subjects_trials = []
        for subject_id_name, trials in subjects_trials_dataset.items(): # subjects
            for trial, data in trials.items(): # trials
                self.features.append(torch.tensor(data.loc[:,features_name].values, dtype=torch.float32).to(device))
                self.labels.append(torch.tensor(data.loc[:,labels_name].values, dtype=torch.float32).to(device))
                self.subjects_trials.append(subject_id_name+trial)

        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        if self.transform:
            feature = self.transform(feature)
            
        if self.label_transform:
            label = self.label_transform(label)

        return feature, label

    def __len__(self):
        return len(self.features)

    def get_subjects_trials():
        return self.subjects_trials



def load_motiondata(dataset_dict, batch_size, num_workers, features_name,labels_name=['R_KNEE_MOMENT_X'],device='cpu',**kwargs):
    n_labels = len(labels_name)
    dataset = MotionDataset(dataset_dict, features_name, labels_name, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, **kwargs)

    return dataloader, n_labels


if __name__=='__main__':
    data_file = os.path.join(const.DATA_PATH,'selection/double_leg_norm_landing_data.hdf5')
    features_name = const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS)
    labels_name = ['R_KNEE_MOMENT_X']
    dataset = MotionDataset(data_file, features_name, labels_name)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    xx=next(iter(dataloader))

    pdb.set_trace()

    model = MLNNBackbone()

    model()

