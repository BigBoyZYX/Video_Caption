import os
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

# Import Path,Vocabulary, utility, evaluator and datahandler module
from config import Path
from dictionary import Vocabulary
from utils import Utils
from evaluate import Evaluator
from data import DataHandler

import random
import numpy as np
import copy

#set seed for reproducibility
utils = Utils()
utils.set_seed(1)


#Import configuration and model

from config import ConfigRecNet
from models.RecNet.model import RecNet

#create Mean pooling object
cfg = ConfigRecNet()
# specifying the dataset in configuration object from {'msvd','msrvtt'}
cfg.dataset = 'msrvtt'
cfg.appearance_feature_extractor = 'inceptionresnetv2' #'resnet101'
#creation of path object
path = Path(cfg,os.getcwd())

#Changing the hyperparameters in configuration object
cfg.batch_size = 100 #training batch size
cfg.n_layers = 1    # number of layers in decoder rnn
cfg.decoder_type = 'lstm'  # from {'lstm','gru'}


#Vocabulary object,
voc = Vocabulary(cfg)
#If vocabulary is already saved or downloaded the saved file
voc.load() #comment this if using vocabulary for the first time or with no saved file

min_count = 5 #remove all words below count min_count
voc.trim(min_count=min_count)
print('Vocabulary Size : ',voc.num_words)
#print('Vocabulary Size : ',voc.num_words)


# Datasets and dataloaders
data_handler = DataHandler(cfg,path,voc)
train_dset,val_dset,test_dset = data_handler.getDatasets()
train_loader,val_loader,test_loader = data_handler.getDataloader(train_dset,val_dset,test_dset)

#Model object
model = RecNet(voc,cfg,path)
#Evaluator object on test data
test_evaluator_greedy = Evaluator(model,test_loader,path,cfg,data_handler.test_dict)
test_evaluator_beam = Evaluator(model,test_loader,path,cfg,data_handler.test_dict,decoding_type='beam')

#Training Loop
cfg.encoder_lr = 1e-4
cfg.decoder_lr = 1e-4
cfg.teacher_forcing_ratio = 1.0
cfg.training_stage = 2
cfg.lmda = 0.2
model.update_hyperparameters(cfg)
# lr_scheduler = ReduceLROnPlateau(model.dec_optimizer, mode='min', factor=cfg.lr_decay_gamma,
#                                      patience=cfg.lr_decay_patience, verbose=True)
for e in range(1,2501):
    lloss_train, recloss_train = model.train_epoch(train_loader,utils)
    #loss_val = model.train_epoch(val_loader,utils)
    #lr_scheduler.step(loss_train)
    if e%1 == 0 :
        print('Epoch -- >',e,'Likelihood Loss -->',lloss_train,'Reconstruction Loss -->',recloss_train)
        greedy_score = test_evaluator_greedy.evaluate(utils,model,e,lloss_train)
        beam_score = test_evaluator_beam.evaluate(utils,model,e,lloss_train)
        print('greedy :',greedy_score)
        print('beam :',greedy_score)

dataiter = iter(train_loader)
features, targets, mask, max_length,_,motion_feat,object_feat= dataiter.next()
features.size()