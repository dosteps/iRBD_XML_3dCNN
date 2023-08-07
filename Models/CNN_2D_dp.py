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
from Utils.cutmix_3d import CutMixCriterion
from sklearn.metrics import roc_auc_score

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
                 dropout=0,
                 weight_decay=1e-5,
                 num_classes=1,
                 structure='standard',
                 num_fold=None,
                 cutmix_alpha=None):
        super().__init__()
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=False)
        self.valid_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=False)
        self.test_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=False)
        self.train_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=False)
        self.valid_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=False)
        self.test_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=False)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_module = nn.BCEWithLogitsLoss()
        if cutmix_alpha: self.train_loss_module = CutMixCriterion(reduction='mean')
        else: self.train_loss_module = nn.BCEWithLogitsLoss(reduction='mean')
        
        self.save_hyperparameters()
        
        if structure == 'standard': 
            self.conv_layer=self._conv_module_standard(in_channel, n_channel, activation)
            fsize = 256*12*12
        elif structure == 'deep': 
            self.conv_layer=self._conv_module_deep(in_channel, n_channel, activation)
            fsize = 512*3*3
        elif structure == 'shallow': 
            self.conv_layer=self._conv_module_shallow(in_channel, n_channel, activation)
            fsize = 128*28*28
        fc1 = nn.Linear(fsize, fc_channel)
        fc2 = nn.Linear(fc_channel, fc_channel)
        fc3 = nn.Linear(fc_channel, num_classes)
        self.dropout = dropout
        self.drop=nn.Dropout(p=dropout)
        self.optimizer=optimizer
        self.weight_decay=weight_decay
        
        self.fc_module = nn.Sequential(
            fc1,
            getattr(nn, activation)(),
            fc2,
            getattr(nn, activation)(),
            fc3
        )
        
    def _conv_module_shallow(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, n_c, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c, n_c*2, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        return conv_layer
    
    def _conv_module_standard(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, n_c, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c, n_c*2, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c*2, n_c*4, kernel_size=(3, 3), stride=(1, 1), padding=0),
        getattr(nn, activation)(),
        nn.Conv2d(n_c*4, n_c*4, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*4, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        return conv_layer
    
    def _conv_module_deep(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, n_c, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c, n_c*2, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c*2, n_c*4, kernel_size=(3, 3), stride=(1, 1), padding=0),
        getattr(nn, activation)(),
        nn.Conv2d(n_c*4, n_c*4, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*4, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c*4, n_c*8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
        getattr(nn, activation)(),
        nn.Conv2d(n_c*8, n_c*8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
        nn.BatchNorm2d(n_c*8, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c*8, n_c*8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
        getattr(nn, activation)(),
        nn.Conv2d(n_c*8, n_c*8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
        nn.BatchNorm2d(n_c*8, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        return conv_layer
    
    
    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1)
        if self.dropout: out = self.drop(out)
        out = self.fc_module(out)
        # probability distribution over labels
        return out
    
    def training_step(self, train_batch, batch_idx):
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
        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['data'].half()
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
        auc = roc_auc_score(y.cpu(), pred.cpu()) if y.float().mean() > 0 else 0.5 # skip sanity check
        self.test_acc(y_hat.sigmoid(), y)
        self.log('test/loss', avg_loss,  prog_bar=False, sync_dist=True)
        self.log('test_auc', auc,  prog_bar=False, sync_dist=True)
        self.log('test/acc', self.test_acc,  prog_bar=False, sync_dist=True)
        
        self.test_pc(y_hat.sigmoid(), y)
        self.log('test/precision', self.test_pc, on_step=False, on_epoch=True, sync_dist=True)
        self.test_rc(y_hat.sigmoid(), y)
        self.log('test/recall', self.test_rc, on_step=False, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self):
      optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.weight_decay)
      return optimizer
  
## plot architecture

# from torchsummary import summary # custom func
# model_sm = C3D_classic(None, None)
# summary(model_sm, input_size=(3, 120, 120), device="cpu")


#%% LitResnet

import torchvision
from torch.optim.lr_scheduler import OneCycleLR

def create_model():
    pretrain = False
    if pretrain:
        model = torchvision.models.resnet18(pretrained=pretrain, num_classes=1000)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    else: model = torchvision.models.resnet18(pretrained=pretrain, num_classes=1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-4,
                 batch_size = 128,
                 in_channel = None,
                 n_channel = None,
                 fc_channel = None,         
                 stride = None,
                 activation=None,
                 optimizer=None,
                 dropout=None,
                 weight_decay=None,
                 num_classes=1,
                 structure=None,
                 num_fold=None,
                 cutmix_alpha = None):
        super().__init__()
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=False)
        self.valid_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=False)
        self.test_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=False)
        self.train_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=False)
        self.valid_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=False)
        self.test_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=False)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_module = nn.BCEWithLogitsLoss()
        if cutmix_alpha: self.train_loss_module = CutMixCriterion(reduction='mean')
        else: self.train_loss_module = nn.BCEWithLogitsLoss(reduction='mean')
        
        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, train_batch, batch_idx):
        x = train_batch['data'].half()
        y = train_batch['label']
        y_hat = self(x)
        return {'target': y, 'preds': y_hat}
    
    def training_step_end(self, batch_parts):
        outputs = batch_parts['preds']
        target = batch_parts['target'].half()
        loss = self.train_loss_module(outputs, target)
        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.train_acc(batch_parts['preds'].sigmoid(), batch_parts['target'])
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch['data'].half()
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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


  

