#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:36:13 2023

@author: nelab
"""

#%% example commands

#0. Sign in WandB account
import wandb
!wandb login

#1. generate train/test indices to split data in pretraining/finetuning stage
!python Methods/1_split_dataset.py -dt "iRBD"

#2. Log Artifact for train/test indices to WandB
!python Methods/2_data_preparation.py -dt "iRBD" -entity "username"

#3. Training CNN classifier
!python Methods/train_ercd_artifacts.py -pipe "init_train" -model "CNN_3D" -dt "iRBD" -entity "username"

#4. Fine-tuning the CNN classifier
!python Methods/finetune_ercd_artifacts.py -pipe "init_train" -model "CNN_3D" -dt "iRBD" -entity "username"

#5. Evaluate the CNN classifier
!python Methods/finetune_ercd_artifacts.py -pipe "eval_plot" -model "CNN_3D" -dt "iRBD" -entity "username"

#6. Interprete the CNN classifier
!python Methods/finetune_ercd_artifacts.py -pipe "explain" -model "CNN_3D" -dt "iRBD" -entity "username"




#%% example commands for demo dataset (CIFAR10)

#1. generate train/test indices to split data in pretraining/finetuning stage
!python Methods/1_split_dataset.py -dt "demo"

#2. Log Artifact for train/test indices to WandB
!python Methods/2_data_preparation.py -dt "demo" -entity "username"

#3. Training CNN classifier
!python Methods/train_ercd_artifacts.py -pipe "init_train" -model "CNN_2D" -dt "demo" -entity "username"

#4. Fine-tuning the CNN classifier
!python Methods/finetune_ercd_artifacts.py -pipe "init_train" -model "CNN_2D" -dt "demo" -entity "username"

#5. Evaluate the CNN classifier
!python Methods/finetune_ercd_artifacts.py -pipe "eval_plot" -model "CNN_2D" -dt "demo" -entity "username"