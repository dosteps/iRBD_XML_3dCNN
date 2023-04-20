#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 01:18:52 2023

@author: nelab
"""

import os
import scipy.io as scio
import numpy as np
import random
from skimage.util import random_noise
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data_utils 
from Utils.cutmix_3d import CutMixCollator
import platform

import torch
import torchvision
import torchvision.transforms as transforms

if os.cpu_count() <= 4: # if google.colab
    CPUS = os.cpu_count()
    PATH = '/content/ERP3D_source_whole_trials_16'
else:
    if platform.system()=='Windows': CPUS = 0
    else: 
        CPUS = 8        
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    PATH = 'Data/dataset_online/ERP3D_source_whole_trials_16'


#%% iRBD ERCD Dataset
class EEGDataset(Dataset):
    """Posner ERP dataset."""
    def __init__(self, filenames, y_score, indices, types='V1', model_alias_name=None, transform=None):
        """
        Args:
            annot_file (string): Path to the file with annotations (ground truth).
            root_dir (string): Directory with all the inputs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.marks = filenames[indices]
        self.ylabel = y_score[indices]
        self.root_dir = '_'.join([PATH, types])
        self.alias_name = model_alias_name
        self.transform = transform

    def __len__(self):
        return len(self.marks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_name = os.path.join(self.root_dir,
                                self.marks[idx][0])
        data = scio.loadmat(data_name)['X_all']
        if 'baseline_all' not in self.alias_name:
            alias_name_chunk = self.alias_name.split('_')
            TOI = [int(s) for s in alias_name_chunk if s.isdigit()]
            data = np.mean(data[int(TOI[0]/50):int(TOI[1]/50),:], axis=0)[np.newaxis, :]
        
        sample = {'data': data, 'label': self.ylabel[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor_train_normalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mtype):
        self.mtype = mtype
        
    def __call__(self, sample):
        data = torch.tensor(sample['data'], dtype = torch.float32) # D x H x W x C
        label = torch.tensor(sample['label'], dtype = torch.long)
        data_norm = transforms.Normalize([data.mean()],[data.std()])(data)
        if self.mtype == 'CNN_3D':
            data_norm = torch.transpose(data_norm.unsqueeze(-1), 0,-1) # C x H x W x D - Confirmed.
        return {'data': data_norm,
                'label': label}
    
class ToTensor_train_norm_noise(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, nlevel, mtype):
        self.nlevel = nlevel
        self.mtype = mtype
        
    def __call__(self, sample):
        data = torch.tensor(sample['data'], dtype = torch.float32) # D x H x W x C
        label = torch.tensor(sample['label'], dtype = torch.long)
        data_norm = transforms.Normalize([data.mean()],[data.std()])(data)
        if random.random() < 0.5:
            gauss_img = torch.tensor(random_noise(data_norm, mode='gaussian', mean=0, var=self.nlevel, clip=True))
        else: gauss_img = data_norm
        if self.mtype == 'CNN_3D':
            gauss_img = torch.transpose(gauss_img.unsqueeze(-1), 0,-1) # C x H x W x D - Confirmed.
        return {'data': gauss_img,
                'label': label}
    
class ToTensor_normalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mtype):
        self.mtype = mtype

    def __call__(self, sample):
        data = torch.tensor(sample['data'], dtype = torch.float32) # D x H x W x C
        label = torch.tensor(sample['label'], dtype = torch.long)
        data_norm = transforms.Normalize([data.mean()],[data.std()])(data)
        if self.mtype == 'CNN_3D':
            data_norm = torch.transpose(data_norm.unsqueeze(-1), 0,-1) # C x H x W x D - Confirmed.
        return {'data': data_norm,
                'label': label}

    
class EEGDataModule_pt(pl.LightningDataModule): # Pretraining dataset
    def __init__(self, filenames, y_score, indices, model_alias_name, config):
        super().__init__()
        self.filenames = filenames
        self.ylabel = y_score
        self.indices = indices
        self.alias_name = model_alias_name
        self.batch_size = config.batch_size
        self.cutmix_alpha = config.cutmix_alpha
        self.vf = config.vf
        self.nlevel = config.nlevel
        self.mtype = config.model_type
        self.cinn = config.num_infold

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if 'verify-noise' in self.vf: tv_transforms = ToTensor_train_norm_noise(self.nlevel, self.mtype)
            else: tv_transforms = ToTensor_train_normalize(self.mtype)
            self.train_dataset = EEGDataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['pt_train'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=tv_transforms
                                  )
            self.valid_dataset = EEGDataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['pt_valid'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=ToTensor_normalize(self.mtype)
                                  )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":            
            self.test_dataset = EEGDataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['eval'],
                                  model_alias_name = self.alias_name,
                                  transform=ToTensor_normalize(self.mtype)
                                  )
    def train_dataloader(self):
        if self.cutmix_alpha:
            collator = CutMixCollator(self.cutmix_alpha)
        else:
            collator = data_utils.dataloader.default_collate
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers = CPUS, pin_memory=True, collate_fn=collator)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                          drop_last=True, num_workers = CPUS, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          drop_last=False, num_workers = CPUS, pin_memory=False)



class EEGDataModule_ft(pl.LightningDataModule): # Finetuning dataset
    def __init__(self, filenames, y_score, indices, model_alias_name, config):
        super().__init__()
        self.filenames = filenames
        self.ylabel = y_score
        self.indices = indices
        self.alias_name = model_alias_name
        self.batch_size = config.batch_size
        self.cutmix_alpha = config.cutmix_alpha
        self.vf = config.vf
        self.nlevel = config.nlevel
        self.mtype = config.model_type
        self.cinn = config.num_infold

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            tv_transforms = ToTensor_train_normalize(self.mtype)
            self.train_dataset = EEGDataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['ft_train'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=tv_transforms
                                  )
            self.valid_dataset = EEGDataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['ft_valid'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=ToTensor_normalize(self.mtype)
                                  )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":            
            self.test_dataset = EEGDataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['eval'],
                                  model_alias_name = self.alias_name,
                                  transform=ToTensor_normalize(self.mtype)
                                  )
    def train_dataloader(self):
        if self.cutmix_alpha:
            collator = CutMixCollator(self.cutmix_alpha)
        else:
            collator = data_utils.dataloader.default_collate
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers = CPUS, pin_memory=True, collate_fn=collator)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False,
                          drop_last=False, num_workers = CPUS, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          drop_last=False, num_workers = CPUS, pin_memory=False)




#%% CIFAR10 Dataset
class CIFAR_DataModule_pt(pl.LightningDataModule): # Pretraining dataset
    def __init__(self, filenames, y_score, indices, model_alias_name, config):
        super().__init__()
        self.filenames = filenames
        self.ylabel = y_score
        self.indices = indices
        self.alias_name = model_alias_name
        self.batch_size = config.batch_size
        self.cutmix_alpha = config.cutmix_alpha
        self.vf = config.vf
        self.nlevel = config.nlevel
        self.mtype = config.model_type
        self.cinn = config.num_infold
        if self.mtype == 'CNN_3D': raise NameError('CNN_3D for CIFAR10 is not supported yet.')
        if config.model_name == 'C3D_classic': self.input_shape = (120,120)
        elif config.model_name == 'LitResnet': self.input_shape = (224,224)
        

    def setup(self, stage=None):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.input_shape),
            transforms.Normalize((0.5), (0.5))])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.Resize(self.input_shape),
            transforms.Normalize((0.5), (0.5))])
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR_Dataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['pt_train'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=train_transforms,
                                  nlevel=self.nlevel
                                  )
            self.valid_dataset = CIFAR_Dataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['pt_valid'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=test_transforms
                                  )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":            
            self.test_dataset = CIFAR_Dataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['eval'],
                                  model_alias_name = self.alias_name,
                                  transform=test_transforms
                                  )
    def train_dataloader(self):
        if self.cutmix_alpha:
            collator = CutMixCollator(self.cutmix_alpha)
        else:
            collator = data_utils.dataloader.default_collate
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers = CPUS, pin_memory=True, collate_fn=collator)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                          drop_last=True, num_workers = CPUS, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          drop_last=False, num_workers = CPUS, pin_memory=False)




class CIFAR_DataModule_ft(pl.LightningDataModule): # Finetuning dataset
    def __init__(self, filenames, y_score, indices, model_alias_name, config):
        super().__init__()
        self.filenames = filenames
        self.ylabel = y_score
        self.indices = indices
        self.alias_name = model_alias_name
        self.batch_size = config.batch_size
        self.cutmix_alpha = config.cutmix_alpha
        self.vf = config.vf
        self.nlevel = config.nlevel
        self.mtype = config.model_type
        self.cinn = config.num_infold
        if self.mtype == 'CNN_3D': raise NameError('CNN_3D for CIFAR10 is not supported yet.')
        if config.model_name == 'C3D_classic': self.input_shape = (120,120)
        elif config.model_name == 'LitResnet': self.input_shape = (224,224)
        

    def setup(self, stage=None):
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.input_shape),
            transforms.Normalize((0.5), (0.5))])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.Resize(self.input_shape),
            transforms.Normalize((0.5), (0.5))])
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CIFAR_Dataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['ft_train'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=train_transforms,
                                  nlevel=self.nlevel
                                  )
            self.valid_dataset = CIFAR_Dataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['ft_valid'][self.cinn],
                                  model_alias_name = self.alias_name,
                                  transform=test_transforms
                                  )
        # Assign test dataset for use in dataloader(s)
        if stage == "test":            
            self.test_dataset = CIFAR_Dataset(filenames=self.filenames,
                                  y_score=self.ylabel,
                                  indices = self.indices['eval'],
                                  model_alias_name = self.alias_name,
                                  transform=test_transforms
                                  )
    def train_dataloader(self):
        if self.cutmix_alpha:
            collator = CutMixCollator(self.cutmix_alpha)
        else:
            collator = data_utils.dataloader.default_collate
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          drop_last=True, num_workers = CPUS, pin_memory=True, collate_fn=collator)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False,
                          drop_last=False, num_workers = CPUS, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1,
                          drop_last=False, num_workers = CPUS, pin_memory=False)


class CIFAR_Dataset(Dataset):
    """CIFAR10 dataset."""
    def __init__(self, filenames, y_score, indices, model_alias_name='', transform=None, nlevel=None):
        """
        Args:
            annot_file (string): Path to the file with annotations (ground truth).
            root_dir (string): Directory with all the inputs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.marks = filenames[indices]
        self.ylabel = y_score[indices]
        self.alias_name = model_alias_name
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR10(root='./Data', train=True,
                                                download=False, transform=transform)
        self.nlevel = nlevel

    def __len__(self):
        return len(self.marks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        data, _ = self.dataset[self.marks[idx]]
        if 'baseline_all' in self.alias_name:
            data = torch.stack([data for i in range(0,16)], dim=3) #[1, 120, 120, 16]
        if self.nlevel != None:
            if random.random()<0.5:
                data = torch.tensor(random_noise(data, mode='gaussian', mean=0, var=self.nlevel, clip=True))
        sample = {'data': data, 'label': self.ylabel[idx]}
        return sample
    
    
