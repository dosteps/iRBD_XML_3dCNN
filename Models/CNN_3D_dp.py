#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:41:57 2022

@author: nelab
"""

import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from sklearn.metrics import roc_auc_score
from Utils.cutmix_3d import CutMixCriterion

GPUS = torch.cuda.device_count()

class C3D_classic(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=128,
                 in_channel=1,
                 n_channel=64,
                 fc_channel=512,
                 stride=2,
                 activation='ReLU',
                 optimizer='AdamW',
                 weight_decay=1e-5,
                 dropout=0,
                 num_classes=1,
                 structure='standard',
                 num_fold=None,
                 cutmix_alpha=0):
        super().__init__()
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_pc = torchmetrics.Precision(average='macro', num_classes=1, multiclass=False)
        self.valid_pc = torchmetrics.Precision(average='macro', num_classes=1, multiclass=False)
        self.test_pc = torchmetrics.Precision(average='macro', num_classes=1, multiclass=False)
        self.train_rc = torchmetrics.Recall(average='macro', num_classes=1, multiclass=False)
        self.valid_rc = torchmetrics.Recall(average='macro', num_classes=1, multiclass=False)
        self.test_rc = torchmetrics.Recall(average='macro', num_classes=1, multiclass=False)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.fc_channel = fc_channel
        
        self.dropout = dropout
        self.drop=nn.Dropout(p=dropout)
        self.optimizer=optimizer
        self.weight_decay=weight_decay
        
        self.loss_module = nn.BCEWithLogitsLoss(reduction='mean')
        if cutmix_alpha: self.train_loss_module = CutMixCriterion(reduction='mean')
        else: self.train_loss_module = nn.BCEWithLogitsLoss(reduction='mean')
        
        if structure == 'standard': 
            self.conv_layer=self._conv_module_standard(in_channel, n_channel, activation)
            fsize = n_channel*4*12*12*1
        elif structure == 'deep': 
            self.conv_layer=self._conv_module_deep(in_channel, n_channel, activation)
            fsize = n_channel*8*3*3*1
        elif structure == 'shallow': 
            self.conv_layer=self._conv_module_shallow(in_channel, n_channel, activation)
            fsize = n_channel*2*28*28*2
        fc1 = nn.Linear(fsize, 512)
        fc2 = nn.Linear(512, 512)
        fc3 = nn.Linear(512, 1)
        self.fc_module = nn.Sequential(
            fc1,
            getattr(nn, activation)(),
            fc2,
            getattr(nn, activation)(),
            fc3
        )
        
    def _conv_module_shallow(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, n_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        nn.Conv3d(n_c, n_c*2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        return conv_layer
    
    def _conv_module_standard(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, n_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        nn.Conv3d(n_c, n_c*2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        nn.Conv3d(n_c*2, n_c*4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        getattr(nn, activation)(),
        nn.Conv3d(n_c*4, n_c*4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c*4, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        return conv_layer
    
    def _conv_module_deep(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, n_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        nn.Conv3d(n_c, n_c*2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        nn.Conv3d(n_c*2, n_c*4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        getattr(nn, activation)(),
        nn.Conv3d(n_c*4, n_c*4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        nn.BatchNorm3d(n_c*4, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        nn.Conv3d(n_c*4, n_c*8, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding='same'),
        getattr(nn, activation)(),
        nn.Conv3d(n_c*8, n_c*8, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding='same'),
        nn.BatchNorm3d(n_c*8, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        nn.Conv3d(n_c*8, n_c*8, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding='same'),
        getattr(nn, activation)(),
        nn.Conv3d(n_c*8, n_c*8, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding='same'),
        nn.BatchNorm3d(n_c*8, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
        )
        return conv_layer
    
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        if self.dropout: out = self.drop(out)
        out = self.fc_module(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
        # print(train_batch)
        x = train_batch['data'].half()
        y = train_batch['label']
        y_hat = self(x)
        return {'target': y, 'preds': y_hat}
    
    def training_step_end(self, batch_parts):
        outputs = batch_parts['preds']
        target = batch_parts['target'].half()
        loss = self.train_loss_module(outputs, target)
        self.train_acc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.train_pc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.train_rc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/precision', self.train_pc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/recall', self.train_rc, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['data']
        y = val_batch['label']
        y_hat = self(x)
        return {'target': y, 'preds': y_hat}
            
    def validation_step_end(self, batch_parts):
        outputs = batch_parts['preds']
        target = batch_parts['target'].half()
        val_loss = self.loss_module(outputs, target)
        #update and log
        self.log('valid/loss', val_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_acc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.log('valid/acc', self.valid_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        self.valid_pc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.log('valid/precision', self.valid_pc, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_rc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.log('valid/recall', self.valid_rc, on_step=False, on_epoch=True, sync_dist=True)
        
        
    def test_step(self, test_batch, batch_idx):
        x = test_batch['data']
        y = test_batch['label']
        logits = self(x)
        return {'y': y.detach(), 'y_hat': logits.detach()}
    
    def test_epoch_end(self, outputs): 
        if len(outputs[0]['y'].shape)==0:
            y = torch.cat([x['y'].unsqueeze(-1) for x in outputs], dim=0)                
            y_hat = torch.cat([x['y_hat'].unsqueeze(-1) for x in outputs], dim=0)
        else:
            y = torch.cat([x['y'] for x in outputs], dim=0)                
            y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        
        avg_loss = self.loss_module(y_hat, y.half())
        pred = y_hat.sigmoid()>0.5
        self.test_acc(y_hat.sigmoid(), y)
        self.log('test/loss', avg_loss,  prog_bar=False, sync_dist=True)
        self.log('test/acc', self.test_acc,  prog_bar=False, sync_dist=True)
        
        self.test_pc(y_hat.sigmoid(), y)
        self.log('test/precision', self.test_pc, on_step=False, on_epoch=True, sync_dist=True)
        self.test_rc(y_hat.sigmoid(), y)
        self.log('test/recall', self.test_rc, on_step=False, on_epoch=True, sync_dist=True)
    
        if int(y.cpu().sum().item()) != len(y.cpu()): # not only one-class for y_true
            auc = roc_auc_score(y.cpu(), pred.cpu()) if y.float().mean() > 0 else 0.5 # skip sanity check
            self.log('test_auc', auc,  prog_bar=False, sync_dist=True)
    
    
    def configure_optimizers(self):
      optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.weight_decay)
      return optimizer

  
## plot architecture

# from torchsummary import summary # custom func
# model_sm = C3D_classic(None, None)
# summary(model_sm, input_size=(1, 120, 120, 16), device="cpu")
