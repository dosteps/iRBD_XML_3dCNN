#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 05:27:06 2022

@author: nelab
"""

import os
import wandb
import torch
import numpy as np
from easydict import EasyDict as edict
import uuid
import shutil

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import sys
import platform
import argparse
import warnings
import skimage
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


#%% Initialize Model
def build_model_and_log(train_config, model_config, model_name=None, alias_name=None):
    with wandb.init(project=PROJECT, group='artifact_log', job_type='initialize', config=model_config, entity=ENTITY) as run:
        config = wandb.config
        folder_name = train_config.mark_label
        
        model = eval(model_name)(**config)
        model_artifact = wandb.Artifact(
            '_'.join([data_model_name, folder_name, 'initialize']), type='model',
            description=model_name,
            metadata=dict(config))
        
        torch.save(model.state_dict(), os.path.join('_temp', 'initialized_model_'+model_config.structure+'.pth'))
        model_artifact.add_file(os.path.join('_temp', 'initialized_model_'+model_config.structure+'.pth'))
        run.log_artifact(model_artifact, aliases=alias_name)
            
        del model
        

#%% Training
def train_and_log(train_config, model_config=None, model_name=None, model_alias_name=None):
    with wandb.init(project=PROJECT,
                    group=data_model_name,
                    name=train_config.num_fold+'fold-'+str(train_config.num_infold+1)+'sfold',
                    tags=[model_alias_name],
                    job_type='train',
                    config=train_config,
                    entity=ENTITY) as run:
        
        if model_config: mtype = model_config.structure
        else: mtype = 'standard'
        
        hash_id = '_temp/'+uuid.uuid4().hex
        os.mkdir(hash_id)
        
        config = run.config
        num_fold = config.num_fold
        num_infold = config.num_infold
        folder_name = config.mark_label
        
        cpfname = 'trained_model_'+num_fold+'fold-'+str(num_infold+1)+'subfold'
        
        fname = 'mark_'+folder_name+'.npy'
        if folder_name == 'R_G_NC_RBDV1_49': data_label = '47513'
        elif folder_name == 'CIFAR_10': data_label = '50000'
        
        # get indices
        idx_at = run.use_artifact('inat_split_indices_'+folder_name+':latest', type='indices')
        idx_dir = idx_at.download(os.path.join(ARTIFACT,'inat_split_indices_'+folder_name))
        indices = np.load(idx_dir+'/'+'_'.join(['idx', num_fold, 'fold.npy']), allow_pickle = True).item()
        
        # get labels
        label_at = run.use_artifact('inat_labels:latest', type='label')
        # download locally
        label_dir = label_at.download(os.path.join(ARTIFACT, 'inat_labels'))
        y_score = np.load(os.path.join(label_dir, fname), allow_pickle=True)
        
        # get initialized model
        model_artifact = run.use_artifact(
            '_'.join([data_model_name, folder_name, 'initialize'])+':'+model_alias_name)
        model_dir = model_artifact.download(os.path.join(ARTIFACT, data_model_name, folder_name))
        model_config = model_artifact.metadata
        model_config['num_fold'] = num_fold
        model_config['structure'] = mtype
        model = eval(model_name)(**model_config)
        model_path = os.path.join(model_dir, "initialized_model_"+model_config['structure']+".pth")
        model.load_state_dict(torch.load(model_path))
        config.update(model_config)
        
        data = run.use_artifact('inat_preproc_data_'+data_label+':latest')
        data_dir = data.download(os.path.join(ARTIFACT, 'inat_preproc_data_'+data_label))
        filenames = np.load(os.path.join(data_dir, 'preproc_data_header.npy'), allow_pickle=True)
        
        if folder_name == 'CIFAR_10':
            dm = CIFAR_DataModule_pt(filenames, y_score, indices, model_alias_name, config)
        if folder_name == 'R_G_NC_RBDV1_49':
            dm = EEGDataModule_pt(filenames, y_score, indices, model_alias_name, config)
        dm.setup('fit')
        
        wandb_logger = WandbLogger()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath = hash_id,
                                                            filename = cpfname,
                                                          save_top_k=1, monitor='valid/acc', mode='max')
        if config.es_counter:
            early_stop_callback = EarlyStopping(monitor="valid/acc", patience=config.es_counter, mode='max', verbose=True)
            callbacks = [checkpoint_callback, early_stop_callback]
        else: callbacks = checkpoint_callback
        
        if GPUS > 1:
            trainer = pl.Trainer(gpus = config.gpus,
                                 auto_lr_find = False,
                                 auto_scale_batch_size = False,
                                 max_epochs = config.epochs,
                                 precision=16 if config.gpus else 32,
                                 logger=wandb_logger,
                                 callbacks=callbacks,
                                 strategy='dp')
        else:
            trainer = pl.Trainer(gpus = config.gpus,
                                 auto_lr_find = False,
                                 auto_scale_batch_size = False,
                                 max_epochs = config.epochs,
                                 precision=16 if config.gpus else 32,
                                 logger=wandb_logger,
                                 callbacks=callbacks,
                                 detect_anomaly=False)
        
        if not 'fixed-lr' in VER:
            lr_log = dict()
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model, datamodule=dm, num_training=100)
            # Pick point based on plot, or get suggestion
            lr_log['lr_optim'] = lr_finder.suggestion()
            # update hparams of the model
            model.hparams.lr = lr_log['lr_optim']
            config.update(lr_log)
            print('optimal learning rate:'+str(lr_log['lr_optim']))
        
        trainer.fit(model, datamodule=dm)
        
        if not 'log' in VER:
            model_artifact = wandb.Artifact(
                '_'.join([data_model_name, folder_name, 'pretrained']), type='model',
                    description='_'.join([data_model_name, num_fold+'fold', 'pretraining stage']),
                    metadata=dict(model_config))
            if config.es_counter: 
                model = eval(model_name).load_from_checkpoint(os.path.join(hash_id, cpfname+'.ckpt'))
            torch.save(model.state_dict(), os.path.join(hash_id, cpfname+'.ckpt'))
            model_artifact.add_file(os.path.join(hash_id, cpfname+'.ckpt'))
            run.log_artifact(model_artifact, aliases=['latest', '_'.join([model_alias_name, cpfname])])
            
        wandb.alert(title="Training Finished", text=data_model_name+'_'+run.name)
        shutil.rmtree(hash_id)
        del model
        

#%% Interpretation
def interpretation(train_config, model_name=None, model_alias_name=None, save_path=None):
    with wandb.init(project=PROJECT,
                    group=data_model_name,
                    name=train_config.num_fold+'fold-'+str(train_config.num_infold+1)+'sfold',
                    tags=[model_alias_name],
                    job_type='interpretation',
                    config=train_config,
                    entity=ENTITY) as run:
        
        config = run.config
        num_fold = config.num_fold
        num_infold = config.num_infold
        folder_name = config.mark_label
        
        fname = 'mark_'+folder_name+'.npy'
        if folder_name == 'R_G_NC_RBDV1_49': data_label = '47513'
        elif folder_name == 'CIFAR_10': data_label = '50000'
        
        # get indices
        idx_at = run.use_artifact('inat_split_indices_'+folder_name+':latest', type='indices')
        idx_dir = idx_at.download(os.path.join(ARTIFACT,'inat_split_indices_'+folder_name))
        indices = np.load(idx_dir+'/'+'_'.join(['idx', num_fold, 'fold.npy']), allow_pickle = True).item()
        
        # get labels
        label_at = run.use_artifact('inat_labels:latest', type='label')
        # download locally
        label_dir = label_at.download(os.path.join(ARTIFACT, 'inat_labels'))
        y_score = np.load(os.path.join(label_dir, fname), allow_pickle=True)
         
        # get model
        cpfname_pt = 'trained_model_'+num_fold+'fold-'+str(num_infold+1)+'subfold'
        alias_name = '_'.join([model_alias_name, cpfname_pt])
        model_artifact = run.use_artifact(
            '_'.join([data_model_name, folder_name, 'pretrained'])+':'+alias_name)
        model_dir = model_artifact.download(os.path.join(ARTIFACT, data_model_name, folder_name))
        model_path = os.path.join(model_dir, cpfname_pt+'.ckpt')
        model_config = model_artifact.metadata
        model = eval(model_name)(**model_config)
        model.load_state_dict(torch.load(model_path))
        config.update(model_config)
        model.eval()
        
        data = run.use_artifact('inat_preproc_data_'+data_label+':latest')
        data_dir = data.download(os.path.join(ARTIFACT, 'inat_preproc_data_'+data_label))
        filenames = np.load(os.path.join(data_dir, 'preproc_data_header.npy'), allow_pickle=True)
        
        if folder_name == 'CIFAR_10':
            dm = CIFAR_DataModule_pt(filenames, y_score, indices, model_alias_name, config)
        if folder_name == 'R_G_NC_RBDV1_49':
            dm = EEGDataModule_pt(filenames, y_score, indices, model_alias_name, config)
        dm.setup('test')
        
        # delete unnecessary structures
        del model._modules['train_acc']
        del model._modules['valid_acc']
        del model._modules['test_acc']
        del model._modules['train_pc']
        del model._modules['valid_pc']
        del model._modules['test_pc']
        del model._modules['train_rc']
        del model._modules['valid_rc']
        del model._modules['test_rc']
        del model._modules['loss_module']
        if 'train_loss_module' in model._modules: del model._modules['train_loss_module']
        
        Relevance_TP = []
        Relevance_FP = []
        Relevance_TN = []
        Relevance_FN = []
        preds = []
        y_test = []
        for ti, sample_batched in enumerate(dm.test_dataloader()):
            test_X = sample_batched['data']
            test_y = sample_batched['label']
            output = model(test_X)
            pred = output.sigmoid()>0.5
            
            if config.attribution == 'LRP_Composite_gamma':
                for i in range(0, len(model._modules['conv_layer'])):
                    model._modules['conv_layer'][i].rule = GammaRule()
                for i in range(0, 2):
                    model._modules['fc_module'][i].rule = EpsilonRule()
                for i in range(2, len(model._modules['fc_module'])):
                    model._modules['fc_module'][i].rule = EpsilonRule(epsilon=0)
                if config.dropout != 0: 
                    model._modules['drop'].rule = IdentityRule()
                    warnings.warn('dropout layer may cause problem when using WandB')
                lrp = LRP(model)
                attribution = lrp.attribute(test_X, target=0)
            elif config.attribution == 'LRP_low_level':
                lrp_low = LRP_2D(model)
                attribution = lrp_low.attribute(test_X, test_y)
            elif config.attribution == 'GuidedGradCam':
                guided_gc = GuidedGradCam(model, model._modules['conv_layer'][4])
                attribution = guided_gc.attribute(test_X, target=0)
            elif config.attribution == 'GuidedGradCam_lastconv':
                guided_gc = GuidedGradCam(model, model._modules['conv_layer'][-1])
                attribution = guided_gc.attribute(test_X, target=0)
            elif config.attribution == 'GradCam':
                layer_gc = LayerGradCam(model, model._modules['conv_layer'][4])
                attribution = layer_gc.attribute(test_X, target=0)
                
            if config.model_type == 'CNN_3D':
                Rel = attribution[0,0,:,:,:].detach().cpu().data.numpy()
                if Rel.shape != (120,120,16): Rel = skimage.transform.resize(Rel,[120,120,16])
            else:
                Rel = attribution[0,0,:,:].detach().cpu().data.numpy()
                if Rel.shape != (120,120): Rel = skimage.transform.resize(Rel,[120,120])
            
            if pred:
                if test_y == 1: # TRUE POSITIVE
                    # print('yes')
                    Relevance_TP.append(Rel)
                else:
                    Relevance_FP.append(Rel)
            else:
                if test_y == 0: # TRUE NEGATIVE
                    Relevance_TN.append(Rel)
                else:
                    Relevance_FN.append(Rel)
                
            preds.append(pred.detach().cpu().data.numpy())
            y_test.append(np.array(test_y[0].detach().cpu()))
            
        # save outputs
        np.save(save_path+'/'+'relevanceTP_'+run.name+'.npy',Relevance_TP,allow_pickle=True)
        np.save(save_path+'/'+'relevanceTN_'+run.name+'.npy',Relevance_TN,allow_pickle=True)
        np.save(save_path+'/'+'relevanceFP_'+run.name+'.npy',Relevance_FP,allow_pickle=True)
        np.save(save_path+'/'+'relevanceFN_'+run.name+'.npy',Relevance_FN,allow_pickle=True)
        np.save(save_path+'/'+'test_preds_'+run.name+'.npy',preds,allow_pickle=True)
        np.save(save_path+'/'+'test_class_'+run.name+'.npy',y_test,allow_pickle=True)
        
        wandb.alert(title="Interpretation Finished", text=data_model_name)

    
#%% Evaluation
def Evaluation(train_config, model_name=None, model_alias_name=None, save_path=None):
    with wandb.init(project=PROJECT,
                    group=data_model_name,
                    name=train_config.num_fold+'fold-'+str(train_config.num_infold+1)+'sfold',
                    tags=[model_alias_name],
                    job_type='evaluation',
                    config=train_config,
                    entity=ENTITY) as run:
        
        config = run.config
        num_fold = config.num_fold
        num_infold = config.num_infold
        folder_name = config.mark_label
        
        fname = 'mark_'+folder_name+'.npy'
        if folder_name == 'R_G_NC_RBDV1_49': data_label = '47513'
        elif folder_name == 'CIFAR_10': data_label = '50000'
        
        # get indices
        idx_at = run.use_artifact('inat_split_indices_'+folder_name+':latest', type='indices')
        idx_dir = idx_at.download(os.path.join(ARTIFACT,'inat_split_indices_'+folder_name))
        indices = np.load(idx_dir+'/'+'_'.join(['idx', num_fold, 'fold.npy']), allow_pickle = True).item()
        
        # get labels
        label_at = run.use_artifact('inat_labels:latest', type='label')
        # download locally
        label_dir = label_at.download(os.path.join(ARTIFACT, 'inat_labels'))
        y_score = np.load(os.path.join(label_dir, fname), allow_pickle=True)
         
        # get model
        cpfname_pt = 'trained_model_'+num_fold+'fold-'+str(num_infold+1)+'subfold'
        alias_name = '_'.join([model_alias_name, cpfname_pt])
        model_artifact = run.use_artifact(
            '_'.join([data_model_name, folder_name, 'pretrained'])+':'+alias_name)
        model_dir = model_artifact.download(os.path.join(ARTIFACT, data_model_name, folder_name))
        model_path = os.path.join(model_dir, cpfname_pt+'.ckpt')
        model_config = model_artifact.metadata
        model = eval(model_name)(**model_config)
        model.load_state_dict(torch.load(model_path))
        config.update(model_config)
        model.eval()
        
        data = run.use_artifact('inat_preproc_data_'+data_label+':latest')
        data_dir = data.download(os.path.join(ARTIFACT, 'inat_preproc_data_'+data_label))
        filenames = np.load(os.path.join(data_dir, 'preproc_data_header.npy'), allow_pickle=True)
        
        if folder_name == 'CIFAR_10':
            dm = CIFAR_DataModule_pt(filenames, y_score, indices, model_alias_name, config)
        if folder_name == 'R_G_NC_RBDV1_49':
            dm = EEGDataModule_pt(filenames, y_score, indices, model_alias_name, config)
        dm.setup('test')
        
        preds = []
        y_test = []
        outputs = []
        mean_fpr = np.linspace(0, 1, 100)
        for ti, sample_batched in enumerate(dm.test_dataloader()):
            # break
            test_X = sample_batched['data']
            test_y = sample_batched['label']
            output = model(test_X)
            pred = output.sigmoid()>0.5
            outputs.append(output.sigmoid().detach().cpu().numpy()[0][0])
            y_test.append(test_y[0].detach().cpu().numpy())
            preds.append(int(pred[0][0]))
        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cf_matrix = np.array([[tn, fp],[fn, tp]])
        # AUROC curve
        fpr, tpr, thresholds = roc_curve(y_test, outputs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        np.save(save_path+'/'+'cf_matrix_'+run.name+'.npy',cf_matrix,allow_pickle=True)
        np.save(save_path+'/'+'roc_auc_'+run.name+'.npy',roc_auc,allow_pickle=True)
        np.save(save_path+'/'+'tpr_interp_'+run.name+'.npy',interp_tpr,allow_pickle=True)
        
        wandb.alert(title="Evaluation Finished", text=data_model_name)
        
def visualization(args, save_path=None):
    # aggregate results  
    if args.testsub == 0:
        if args.data_type == 'demo': num_files = 100
        elif args.data_type == 'iRBD': num_files = 98
        tprs = []
        aucs = []
        cf_matrix_all = []
        for ci in range(49, num_files):
            num_fold = str(ci+1)
            num_infold = 0 # inner loop
            name=num_fold+'fold-'+str(num_infold+1)+'sfold.npy'
            cf_matrix = np.load(save_path+'/'+'cf_matrix_'+name,allow_pickle=True)
            interp_tpr = np.load(save_path+'/'+'tpr_interp_'+name,allow_pickle=True)
            roc_auc = np.load(save_path+'/'+'roc_auc_'+name,allow_pickle=True)
            cf_matrix_all.append(cf_matrix)
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
        cf_matrix = np.sum(cf_matrix_all,axis=0)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
    else:
        num_fold = str(args.testsub)
        num_infold = 0 # inner loop
        name=num_fold+'fold-'+str(num_infold+1)+'sfold.npy'
        cf_matrix = np.load(save_path+'/'+'cf_matrix_'+name,allow_pickle=True)
        mean_tpr = np.load(save_path+'/'+'tpr_interp_'+name,allow_pickle=True)
        mean_auc = np.load(save_path+'/'+'roc_auc_'+name,allow_pickle=True)
    
    # Confusion matrix
    if args.data_type == 'demo':
        categories = ['others','terrestrial-animal']
    elif args.data_type == 'iRBD':
        categories = ['NC','iRBD']
    group_names = ['True Negative','False Positive','False Negative','True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         np.divide(cf_matrix, np.sum(cf_matrix,axis=0)).flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    cf_matrix_pct = np.divide(cf_matrix, np.sum(cf_matrix,axis=0))*100
    fig, ax = plt.subplots(figsize=(6.4,4.8))         # Sample figsize in inches
    sns.heatmap(cf_matrix_pct, annot=labels, fmt='',xticklabels=categories, yticklabels=categories, cmap='Blues', ax=ax, annot_kws={"size": 16})
    plt.xlabel('Predicted class', fontsize = 18)
    plt.ylabel('True class', fontsize = 18)
    plt.savefig(os.path.join(save_path, 'confusion_matrix.jpg'))

    # AUC curve
    fig, ax = plt.subplots(figsize=(6.4,4.8))
    mean_fpr = np.linspace(0, 1, 100)
    
    if args.testsub == 0:
        mean_auc = auc(mean_fpr, mean_tpr)
        std_tpr = np.std(tprs, axis=0)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
    else:
        plt.plot(mean_fpr, mean_tpr, color='b',
                label=r'ROC curve (AUC = %0.2f)' % (mean_auc),
                lw=2, alpha=.8)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(os.path.join(save_path, 'ROC_curve.jpg'))


#%% Main
if __name__ == '__main__':
    #### 0. Configuration
    parser = argparse.ArgumentParser(description='Training CNN classifier in pretraining stage')
    parser.add_argument('--pipeline', '-pipe', type=str, default='init_train_explain_eval_plot',
                        help='''set data analysis to perform // init: initialze model, train: train model, 
                        explain: interprete model, eval: evaluate model, plot: plot confusion matrix and ROC curve //
                        example use case: init_train_eval''')
    parser.add_argument('--verify', '-vf', type=str, default='',
                        help='''Evaluate robustness of classifier // verify-structure: test architecture, verify-noise: test noise, 
                        fixed-lr: using fixed learning rate to train, log: log artifact to WandB // 
                        example use case: fixed-lr_verify-structure // 
                        when using '-vf' option, <pipeline: eval_explain> does not support''')
    parser.add_argument('--data-type', '-dt', type=str, default='iRBD',
                        help='type of input data // demo: CIFAR10, iRBD: EEG data')
    parser.add_argument('--user-name', '-entity', type=str, default='nelab',
                        help='WandB account username')
    parser.add_argument('--model-type', '-model', type=str, default='CNN_2D',
                        help='type of CNN classifier // CNN_2D, CNN_3D')
    parser.add_argument('--model-name', '-name', type=str, default='C3D_classic',
                        help='name of CNN classifier // C3D_classic, LitResnet (only applicapable for 2dCNN based on CIFAR10)')
    parser.add_argument('--attribute', '-attr', type=str, default='LRP_Composite_gamma',
                        help='interpretation method for CNN classifier // LRP_Composite_gamma, GuidedGradCam')
    parser.add_argument('--testsub', '-sub', type=str, default='0',
                        help='test single patient (#50~98), ex) "-sub 50"')
    args = parser.parse_args()
    
    sys.path.append(os.getcwd())
    GPUS = torch.cuda.device_count()
    if os.cpu_count() <= 4: # if google.colab
        CPUS = os.cpu_count()
        PATH = '/content/ERP3D_source_whole_trials_16'
    else:
        if platform.system()=='Windows': CPUS = 0
        else: 
            CPUS = 8        
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        PATH = 'Data/dataset_online/ERP3D_source_whole_trials_16'
    """ --------------------------- OPTION -------------------------------"""
    ENTITY = args.user_name # WandB account name
    PROJECT = args.data_type+'_XML_3dCNN' # WandB project name
    PREFIX = args.model_type # 'CNN_2D', 'CNN_3D'
    VER = args.verify # '', 'verify-structure', 'verify-noise'
    model_name = args.model_name
    """-------------------------------------------------------------------"""
    ARTIFACT = 'artifacts'
    if PREFIX == 'CNN_2D':
        from Models.CNN_2D_dp import C3D_classic, LitResnet
        if GPUS > 1: 
            model_alias_name = 'baseline_200_350' # 'baseline_all' 'baseline_200_350' 'baseline_start_end [0~800]'
        elif GPUS == 1: 
            # from Models.CNN_2D import C3D_classic
            model_alias_name = 'sGPU_baseline_200_350' # 'baseline_all' 'baseline_200_350' 'baseline_start_end [0~800]'
        else:
            raise NameError('CPU is not appropriate for deep learning in this script')
    elif PREFIX == 'CNN_3D':
        from Models.CNN_3D_dp import C3D_classic
        if GPUS > 1: 
            model_alias_name = 'baseline_all' # 'baseline_all'
        elif GPUS == 1: 
            warnings.warn('single GPU is not recommended for 3dCNN due to memory limit')
            model_alias_name = 'sGPU_baseline_all' # 'baseline_all'
        else:
            raise NameError('CPU is not appropriate for deep learning in this script')
            
    if VER == '': data_model_name = '_'.join([PREFIX, model_name])
    else: data_model_name = '_'.join([PREFIX, model_name, VER])
    model_config = edict({'learning_rate':1e-4,
                    'batch_size': 128,
                    'in_channel': 1, # channel number of input data
                    'n_channel': 64, # channel unit of convolutional layer
                    'fc_channel': 512, # channel unit of fully-connected layer
                    'stride': 2,
                    'activation':'ReLU',
                    'optimizer':'AdamW',
                    'weight_decay':1e-5,
                    'dropout': 0, # probablity <0~1>
                    'structure':'standard', # 'shallow', 'standard', 'deep'
                    'cutmix_alpha': 0})
    if args.data_type == 'demo':
        from Utils.data import CIFAR_DataModule_pt
        marks2use = 'CIFAR_10' # Demo-dataset
        model_config.in_channel = 3
        if VER == 'verify-noise': raise NameError('it is not implemented to evaluate robustness of classifier for adding noise to input data (CIFAR10)')
    elif args.data_type == 'iRBD':
        from Utils.data import EEGDataModule_pt
        marks2use = 'R_G_NC_RBDV1_49' # file including information for ground-truth
    else:
        raise NameError('invalid data-type')
    train_config = edict({"gpus":GPUS,
                          "model_type":PREFIX,
                          "model_name":model_name,
                          "mark_label":marks2use,
                          "attribution":args.attribute,
                          "vf":VER,
                          "nlevel":0,
                          "epochs": 100,
                          "es_counter": 10}) # if no early stopping, es_counter = None
        
    os.makedirs('_temp', exist_ok=True)
    # savefile path for result to use it in MATLAB
    if 'baseline_all' in model_alias_name: tag = 'whole'
    else:
        alias_name_chunk = model_alias_name.split('_')
        TOI = [int(s) for s in alias_name_chunk if s.isdigit()]
        tag = str(TOI[0])+'_'+str(TOI[1])
    
    result_path = os.path.join('Results',marks2use,'pretrained','_'.join([model_name, train_config.model_type, tag]))
    result_XML_path = os.path.join('Results',marks2use,'pretrained','_'.join([model_name, train_config.model_type, tag]),train_config.attribution)
    os.makedirs(result_path, exist_ok = True)
    os.makedirs(result_XML_path, exist_ok = True)
        
    #### 1. initialize model
    if 'init' in args.pipeline:
        print('[init] Now Processing...initialize classifier')
        if not 'verify-structure' in VER:
            build_model_and_log(train_config, model_config, model_name=model_name, alias_name=model_alias_name) # Run Only at first time
        else:
            model_type = ['shallow', 'standard', 'deep']
            for mtype in model_type:
                model_config.structure = mtype
                build_model_and_log(train_config, model_config, model_name=model_name, alias_name=model_alias_name+'_'+mtype) # Run Only at first time

    #### 2. pre-train model
    if 'train' in args.pipeline:
        print('[train] Now Processing...training classifier')
        if VER == '':
            train_config.num_infold = 0 # inner loop
            if args.testsub == '0':
                if args.data_type == 'demo': num_files = 100
                elif args.data_type == 'iRBD': num_files = 98
                for ci in range(49, num_files):
                    train_config.num_fold = str(ci+1)
                    train_and_log(train_config, model_config=model_config, model_name=model_name, model_alias_name=model_alias_name)
            else:
                sub = args.testsub.split('_')
                if len(sub) > 1: # ex. -sub '60_65' (= patient 60~65)
                    for ci in range(int(sub[0]), int(sub[1])+1):
                        train_config.num_fold = str(ci)
                        train_and_log(train_config, model_config=model_config, model_name=model_name, model_alias_name=model_alias_name)
                else: # ex. -sub '50' (= patient 50)
                    train_config.num_fold = str(args.testsub)
                    train_and_log(train_config, model_config=model_config, model_name=model_name, model_alias_name=model_alias_name)
        elif 'verify-structure' in VER:
            args.testsub = '50'
            train_config.num_fold = str(args.testsub)
            model_type = ['shallow', 'standard', 'deep']
            for cinn in range(0, 10): # 10-fold nested cross-validation
                for mtype in model_type:
                    train_config.num_infold = cinn # inner loop
                    model_config.structure = mtype
                    train_and_log(train_config, model_config=model_config, model_name=model_name, model_alias_name=model_alias_name+'_'+mtype)
        elif 'verify-noise' in VER:
            args.testsub = '50'
            train_config.num_fold = str(args.testsub)
            nlevels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            for cinn in range(0, 1): # 10-fold nested cross-validation
                for nlevel in nlevels:
                    train_config.nlevels = nlevel
                    train_config.num_infold = cinn # inner loop
                    train_and_log(train_config, model_config=None, model_name=model_name, model_alias_name=model_alias_name)
    
    #### 3. explain classifier's decision
    if 'explain' in args.pipeline:        
        print('[explain] Now Processing...interprete classifier')
        if train_config.attribution == 'LRP_low_level':
            from Methods.interpretation import LRP_2D
        elif 'GradCam' in train_config.attribution:
            from captum.attr import GuidedGradCam, LayerGradCam
        else:
            from captum.attr import LRP
            from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, IdentityRule

        if args.testsub == '0':
            if args.data_type == 'demo': num_files = 100
            elif args.data_type == 'iRBD': num_files = 98
            for ci in range(49, num_files):
                train_config.num_fold = str(ci+1)
                train_config.num_infold = 0 # inner loop
                interpretation(train_config, model_name=model_name, model_alias_name=model_alias_name, save_path=result_XML_path)
        else:
            train_config.num_fold = str(args.testsub)
            train_config.num_infold = 0 # inner loop
            interpretation(train_config, model_name=model_name, model_alias_name=model_alias_name, save_path=result_XML_path)
            
    #### 4. evaluate classifier
    if 'eval' in args.pipeline:    
        print('[eval] Now Processing...evaluate classifier')
        if args.testsub == '0':
            if args.data_type == 'demo': num_files = 100
            elif args.data_type == 'iRBD': num_files = 98
            for ci in range(49, num_files):
                train_config.num_fold = str(ci+1)
                train_config.num_infold = 0 # inner loop
                Evaluation(train_config, model_name=model_name, model_alias_name=model_alias_name, save_path=result_path)
        else:
            train_config.num_fold = str(args.testsub)
            train_config.num_infold = 0 # inner loop
            Evaluation(train_config, model_name=model_name, model_alias_name=model_alias_name, save_path=result_path)
    
    #### 5. plot resutls
    if 'plot' in args.pipeline:
        print('[plot] Now Processing...plot result')
        visualization(args, result_path)
        
