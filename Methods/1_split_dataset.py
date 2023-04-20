#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 04:03:15 2022

@author: nelab
"""

import os
import scipy.io as scio
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut
import argparse

parser = argparse.ArgumentParser(description='generate train/test indices to split data in both pretraining and finetuning stage')
parser.add_argument('--data-type', '-dt', type=str, default='iRBD',
                    help='type of input data // demo: CIFAR10, iRBD: EEG data')

args = parser.parse_args()

if args.data_type == 'demo':
    marks2use = 'CIFAR_10' # Demo-dataset
elif args.data_type == 'iRBD':
    marks2use = 'R_G_NC_RBDV1_49' # file including information for ground-truth
else:
    raise NameError('invalid data-type')
    
    
folder_name = marks2use
# generate indices for cross-validation as deterministic way
os.makedirs(os.path.join('Data', 'input'), exist_ok=True)
os.makedirs(os.path.join('Results', 'cv_split_indices_'+folder_name), exist_ok=True)
val_split_ratio = 1/10


if marks2use == 'R_G_NC_RBDV1_49':
    # # Device configuration
    ABS_Path = os.getcwd()
    annot_file = ABS_Path + '/Data/dataset_online/ERP3D_source_whole_trials_16_V1/mark.mat'
    marks = scio.loadmat(annot_file)[marks2use]
    mark = marks[:, 0:3]
elif marks2use == 'CIFAR_10':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    def To_binary_class(target):
        # [3~7]: terrestrial animal
        if 2<target<8: # terrestrial animal (=:iRBD)
            target = 2
        else: # etc. (=:Normal control)
            target = 1
        return target
    trainset = torchvision.datasets.CIFAR10(root='./Data', train=True,
                                            download=True, transform=transforms.ToTensor(),
                                            target_transform=To_binary_class)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                              shuffle=False, num_workers=0)
    _, label = next(iter(trainloader))
    n_sub = np.array([i//500 for i in range(0,50000)]).astype(int)
    nc_idx = label.numpy()==1
    
    mark = np.empty((50000,3), dtype=object)
    mark[:,0] = np.array([i for i in range(0,50000)]).astype(int)
    for i in range(0, len(label)): mark[i,1] = label[i].numpy().reshape(1,1)
    # mark[:,1] = label.numpy().astype(int).shape
    mark[nc_idx,2] = n_sub[:25000]
    mark[~nc_idx,2] = n_sub[25000:]
np.save('Data/input/mark_'+marks2use+'.npy', mark, allow_pickle=True)
    
targets = np.array(mark[:,1]==2)
groups = np.array(mark[:,2].tolist()).flatten()

cv = LeaveOneGroupOut()
for ci, (idx_train, idx_test) in enumerate(cv.split(mark[:,0], targets, groups)):
    y_label = np.unique(mark[idx_test,1][0])
    if len(y_label)>1:
        NameError('check indices')
    else:
        # to extract residuals from opposite class
        if y_label == 1: 
            idx_train_eq = idx_train[mark[idx_train,1]==1]
            idx_train_op = idx_train[mark[idx_train,1]==2]
        else:
            idx_train_eq = idx_train[mark[idx_train,1]==2]
            idx_train_op = idx_train[mark[idx_train,1]==1]
        (idx_train_op_chunk, idx_ftune_residual) = train_test_split(idx_train_op,test_size=len(idx_test),shuffle=True)
        
        # PreTraining phase
        idx_train_concat = np.concatenate((idx_train_eq, idx_train_op_chunk), axis=0)
        marks_trn = mark[idx_train_concat]
        undersample = RandomUnderSampler(sampling_strategy='majority')
        idx_us, y_us = undersample.fit_resample(np.arange(len(marks_trn))[:,np.newaxis], marks_trn[:,1]==2)
        idx_train_us = idx_train_concat[idx_us.squeeze()]
        marks_trn_us = mark[idx_train_us]
        
        ptcv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        pt_targets = np.array(marks_trn_us[:,1]==2)
        pttrn_indices = list()
        ptval_indices = list()
        for _ ,(pttrn_ix, ptval_ix) in enumerate(ptcv.split(idx_train_us, pt_targets)):
            pttrn_idx = idx_train_us[pttrn_ix]
            ptval_idx = idx_train_us[ptval_ix]
            pttrn_indices.append(pttrn_idx)
            ptval_indices.append(ptval_idx)
        
        # FineTuning phase
        idx_test_concat = np.concatenate((idx_test, idx_ftune_residual), axis=0)
        y_test_concat = mark[idx_test_concat][:,1]
        # 80 % fine-tuning, 20 % evaluation
        (idx_test_tune, idx_test_eval) = train_test_split(idx_test_concat, test_size=round(len(idx_test_concat)*0.2), stratify=y_test_concat)
        
        marks_tune = mark[idx_test_tune]
        ftcv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        ft_targets = np.array(marks_tune[:,1]==2)
        fttrn_indices = list()
        ftval_indices = list()
        for _ ,(fttrn_ix, ftval_ix) in enumerate(ftcv.split(idx_test_tune, ft_targets)):
            fttrn_idx = idx_test_tune[fttrn_ix]
            ftval_idx = idx_test_tune[ftval_ix]
            fttrn_indices.append(fttrn_idx)
            ftval_indices.append(ftval_idx)
        
        idx = {}
        idx['pt_train'] = pttrn_indices # 10 subsets
        idx['pt_valid'] = ptval_indices # 10 subsets
        idx['ft_train'] = fttrn_indices # 10 subsets
        idx['ft_valid'] = ftval_indices # 10 subsets
        idx['eval'] = idx_test_eval
        
        
        # Sanity check
        for i in range(0,10):
            if len(set(idx['eval']).intersection(set(idx['pt_train'][i])))!=0:
                raise NameError('invalid split found')
            if len(set(idx['eval']).intersection(set(idx['pt_valid'][i])))!=0:
                raise NameError('invalid split found')
            if len(set(idx['eval']).intersection(set(idx['ft_train'][i])))!=0:
                raise NameError('invalid split found')
            if len(set(idx['eval']).intersection(set(idx['ft_valid'][i])))!=0:
                raise NameError('invalid split found')
        
        np.save(os.path.join('Results','cv_split_indices_'+folder_name,'idx_'+str(ci+1)+'_fold.npy'), idx, allow_pickle=True)