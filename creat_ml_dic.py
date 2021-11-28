from dictionary import Vocabulary
from config import ConfigRecNet
from config import Path
import os
from data import DataHandler

#create Mean pooling object
cfg = ConfigRecNet()
# specifying the dataset in configuration object from {'msvd','msrvtt'}
cfg.dataset = 'ml'
cfg.appearance_feature_extractor = 'inceptionresnetv2' #
#creation of path object
path = Path(cfg,os.getcwd())

voc = Vocabulary(cfg)
# Datasets and dataloaders
data_handler = DataHandler(cfg,path,voc)

voc = Vocabulary(cfg)
dicts = [data_handler.train_dict, data_handler.test_dict, data_handler.val_dict]
for dict in dicts:
    for video_id, captions in dict.items():
        for caption in captions:
            for word in caption.split(' '):
                voc.addWord(word)
voc.save()
