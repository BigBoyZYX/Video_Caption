import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Import Path,Vocabulary, utility, evaluator and datahandler module
from config import Path
from dictionary import Vocabulary
from utils import Utils
from evaluate import Evaluator
from data import DataHandler
#Import configuration and model
from config import ConfigMARN
from models.RecNet.model import RecNet
from models.MARN.model import MARN

#set seed for reproducibility
utils = Utils()
utils.set_seed(1)

mode = 'train'
#create Mean pooling object
cfg = ConfigMARN()
# specifying the dataset in configuration object from {'msvd','msrvtt'}
cfg.dataset = 'ml'
cfg.appearance_feature_extractor = 'pnasnet5large' #inceptionresnetv2
cfg.motion_feature_extractor = 'se_resnext101_32x4d'
cfg.prediction_path = 'ML/test2'
cfg.saved_models_path = 'ML/test2'
#creation of path object
path = Path(cfg,os.getcwd())

epochs=3000

#Changing the hyperparameters in configuration object
cfg.batch_size = 150 #training batch size
cfg.n_layers = 1    # number of layers in decoder rnn
cfg.decoder_type = 'lstm'  # from {'lstm','gru'}

if not os.path.exists(cfg.prediction_path):
    os.mkdir(cfg.prediction_path)

#Vocabulary object,
voc = Vocabulary(cfg)
#If vocabulary is already saved or downloaded the saved file
voc.load() #comment this if using vocabulary for the first time or with no saved file

min_count = 3 #remove all words below count min_count
voc.trim(min_count=min_count)
print('Vocabulary Size : ',voc.num_words)

# Datasets and dataloaders
data_handler = DataHandler(cfg,path,voc)
train_dset,val_dset,test_dset = data_handler.getDatasets()
train_loader,val_loader,test_loader = data_handler.getDataloader(train_dset,val_dset,test_dset)

#Model object
model = MARN(voc,cfg,path)
#Evaluator object on val data
val_evaluator_greedy = Evaluator(model,val_loader,path,cfg,data_handler.val_dict,decoding_type = 'greedy')
val_evaluator_beam = Evaluator(model,val_loader,path,cfg,data_handler.val_dict,decoding_type='beam')

#Training Loop
cfg.encoder_lr = 1e-4
cfg.decoder_lr = 1e-4
cfg.teacher_forcing_ratio = 1.0
cfg.training_stage = 2
cfg.lmda = 0.2
model.update_hyperparameters(cfg)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.dec_optimizer, mode='min', factor=cfg.lr_decay_gamma,
                                      patience=cfg.lr_decay_patience, verbose=True)
best_score = 0
if mode == 'train':
    log_file = open(os.path.join(cfg.saved_models_path, 'log.txt'), 'w')
    for e in range(1,epochs+1):
        lloss_train, recloss_train = model.train_epoch(train_loader,utils)
        loss_val = model.train_epoch(val_loader,utils)
        #lr_scheduler.step(lloss_train + recloss_train)
        if e%30 == 0 :
            print('Epoch[{}], Likelihood Loss:{}, Reconstruction Loss:{}'.format(e,lloss_train,recloss_train))
            log_file.write('Epoch[{}], Likelihood Loss:{}, Reconstruction Loss:{}\n'.format(e,lloss_train,recloss_train))
            socre = val_evaluator_beam.evaluate(utils,model,e,lloss_train)
            score_blue = (socre['Bleu_1'] + socre['Bleu_2'] + socre['Bleu_3'] + socre['Bleu_3'])/4
            score_avg = (score_blue + socre['METEOR']+socre['ROUGE_L']+socre['CIDEr'])/4
            log_file.write("beam score: Avg:{.5f} Blue:{.5f} METEOR:{.5f} ROUGR_L:{.5f} CIDEr:{.5f}".format(\
                score_avg, score_blue, socre['METEOR'], socre['ROUGE_L'], socre['CIDEr']))
            model.save(e)

            if best_score < socre:
                best_score = socre
                log_file.write('best greedy socre: {} at epuch #{}'.format(best_score, e))
                model.save(0,suffix='_best_greedy')
    log_file.close()

elif mode == 'infer':
    model.load(0,suffix='_best_beam')
    model.eval()
    all_name = []
    all_caps = []
    from tqdm import tqdm
    with torch.no_grad():
        for data in tqdm(test_loader):
            features, targets, mask, max_length, names, _, _ = data
            #caption,caps_text, atten_val = model.GreedyDecoding(features.to(cfg.device))
            caps_text = model.BeamDecoding(features.to(cfg.device), width=3)
            all_name += names
            all_caps += caps_text

    print('Write answers in required format...')
    result = {}
    result["predictions"] = {}
    for id, cap in zip(all_name, all_caps):
        result["predictions"][id] = []
        itm = {}
        itm["image_id"] = id
        itm["caption"] = cap
        result["predictions"][id].append(itm)
    import json
    with open('answer.json','w') as f:
        json.dump(result, f)



