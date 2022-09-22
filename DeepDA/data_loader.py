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

def load_data(data_folder, batch_size, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    return data_loader, n_class


def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0



class MotionDataset(Dataset):
    def __init__(self, data_file, features_name, labels_name, transform=None,label_transform=None):
        if(isinstance(data_file,str)):
            subjects_trials_dataset = pro_rd.load_subjects_dataset(h5_file_name = data_file, selected_data_fields=features_name+labels_name)
        elif(isinstance(data_file,dict)):
            subjects_trials_dataset = data_file
        else:
            print('data_file is wrong')
            exit()

        self.features = []
        self.labels = []
        self.subjects_trials = []
        for subject_id_name, trials in subjects_trials_dataset.items():
            for trial, data in trials.items():
                self.features.append(torch.tensor(data.loc[:,features_name].values, dtype=torch.float32))
                self.labels.append(torch.tensor(data.loc[:,labels_name].values, dtype=torch.float32))
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



def load_motiondata(dataset_dict, batch_size, train, num_workers,features_name,labels_name=['R_KNEE_MOMENT_X'],**kwargs):
    n_labels = len(labels_name)
    dataset = MotionDataset(dataset_dict,features_name, labels_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, **kwargs)

    return dataloader, n_labels


if __name__=='__main__':
    data_file = os.path.join(const.DATA_PATH,'kam_norm_landing_data.hdf5')
    features_name = ['TIME'] + const.extract_imu_fields(const.IMU_SENSOR_LIST,const.ACC_GYRO_FIELDS)
    labels_name = ['R_KNEE_MOMENT_X']
    dataset = MotionDataset(data_file,features_name, labels_name)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    xx=next(iter(dataloader))

    pdb.set_trace()

    model = MLNNBackbone()

    model()

