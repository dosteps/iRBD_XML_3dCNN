#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 04:16:40 2022

@author: nelab
"""

import os
import wandb
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='generate train/test indices to split data in both pretraining and finetuning stage')
parser.add_argument('--data-type', '-dt', type=str, default='iRBD',
                    help='type of input data // demo: CIFAR10, iRBD: EEG data')
parser.add_argument('--user-name', '-entity', type=str, default='nelab',
                    help='WandB account username')

args = parser.parse_args()

if args.data_type == 'demo':
    marks2use = 'CIFAR_10' # Demo-dataset
elif args.data_type == 'iRBD':
    marks2use = 'R_G_NC_RBDV1_49' # file including information for ground-truth
else:
    raise NameError('invalid data-type')

do_upload = True # TURN IT OFF WHEN REPRODUCE

ENTITY = args.user_name
PROJECT = args.data_type + '_XML_3dCNN'
PREFIX = 'inat'

#%% Upload data to wandb cloud storage
if do_upload:
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(project=PROJECT, group='artifact_log', job_type='upload', entity=ENTITY) as run:
        folder_name = marks2use
        label_name = '_'.join(['mark', marks2use])
        if args.data_type == 'demo': num_files = 100
        elif args.data_type == 'iRBD': num_files = 98
        comment = "indices for pretraining and fine-tuning stage, training:validation = 9:1 for both stages, fine-tuning:evaluation = 8:2"
        
        ''' 1. split indices '''
        # 🏺 create our Artifact
        splits = wandb.Artifact(
            '_'.join([PREFIX, 'split_indices_'+folder_name]), type="indices",
            description=comment)
    
        for ci in range(0, num_files):
            name = '_'.join(['idx', str(ci+1), 'fold'])
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with splits.new_file(name+'.npy', mode="wb") as file:
                indices = np.load('Results/cv_split_indices_'+folder_name+'/'+name+'.npy', allow_pickle=True).item()
                np.save(file, indices, allow_pickle=True)
        # ✍️ Save the artifact to W&B.
        run.log_artifact(splits)
        
    
        ''' 2. data '''
        # # 🚀 start a run, with a type to label it and a project it can call home
        # Configuration
            
        #### Label
        # 🏺 create our Artifact
        label = wandb.Artifact(
            '_'.join([PREFIX, 'labels']), type='label',
            description='data information',
            metadata={'source': 'marks'})
        
        marks = np.load(os.path.join('Data', 'input', label_name)+'.npy', allow_pickle=True)
        y_total = np.array([marks[i,1][0]==2 for i in range(0, len(marks))]).astype(int)
        with label.new_file(label_name+'.npy', mode="wb") as file:
            np.save(file, y_total, allow_pickle=True)
            del y_total
            
        run.log_artifact(label)
        
        #### DATA
        preproc_data = wandb.Artifact(
            '_'.join([PREFIX, 'preproc', 'data', str(len(marks))]), type='preproc_data',
            description='data information for single-trial ERCD dataset'
            )
        
        filenames = marks[:,0]
        with preproc_data.new_file('_'.join(['preproc', 'data_header.npy']), mode="wb") as file:
            np.save(file, filenames, allow_pickle=True)
        del filenames
        
        run.log_artifact(preproc_data)
            