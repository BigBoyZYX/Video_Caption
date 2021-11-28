import os
import cv2
import numpy as np
import sys
import glob
import json
import h5py
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader

root = os.sep.join(os.getcwd().split(os.sep)[:5]) # may change based on dataset path
save_path = 'Saved_features'
device = torch.device('cuda:0')

# motion [57, 2048] pnasnet5large
# RESNEXT101_64X4D
model_name = 'resnext101_64x4d'#'inceptionresnetv2' # could be fbresnet152 or inceptionresnetv2, se_resnext101_32x4d,pnasnet5large
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model = model.to(device)
model.eval()

data_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((299,299)),transforms.ToTensor(),
                                    transforms.Normalize(model.mean,model.std,inplace=True)])

def video2tensor(video_path,total_frame):
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    frames = []
    fail = 0
    while success:
        # OpenCV Uses BGR Colormap
        success, image = vidObj.read()
        if success:
            RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #transform images
            #print(RGBimage.shape)
            frames.append(data_transform(RGBimage))
            count += 1
        else:
            fail += 1
    vidObj.release()
    frames = torch.stack(frames)
    #frames = torch.from_numpy(frames)
    # take 28 frames per clip uniformly
    interval = count//total_frame
    frames = frames[range(0,interval*total_frame,interval)]
    return frames,count,fail

dset_path = os.path.join(os.path.join(root,'Datasets'),'ML')
ml_train_path = '/home/meihaiwei/HDD/dataset/seu_ml_course/course_dataset/Train/G_15000-G_17249/video/videos'
ml_test_path = '/home/meihaiwei/HDD/dataset/seu_ml_course/course_dataset/Test/video'
ml_train_name_list = glob.glob(ml_train_path+os.sep+'*')
ml_test_name_list = glob.glob(ml_test_path+os.sep+'*')

save_file_train = os.path.join(save_path,'ML_APPEARANCE_RESNEXT101_64X4D.hdf5') #ML_APPEARANCE_INCEPTIONRESNETV2_28
save_file_test = os.path.join(save_path,'ML_TEST_APPEARANCE_RESNEXT101_64X4D.hdf5')

# extra train videos feats
if 1:
    with torch.no_grad():
        with h5py.File(save_file_train, 'w') as f:
            for name in tqdm(ml_train_name_list):
                tensor,_,_ = video2tensor(name,28)
                output_features = model.features(tensor.to(device)).mean(dim=(2,3)) # output_features的第二个维度就是网络输出维度
                ide = name.split(os.sep)[-1].split('.')[0]
                f.create_dataset(ide, data = output_features.cpu().numpy())

#extra test video feats
if 1:
    with torch.no_grad():
        with h5py.File(save_file_test, 'w') as f:
            for name in tqdm(ml_test_name_list):
                tensor,_,_ = video2tensor(name,28) #28, 57 for mention
                output_features = model.features(tensor.to(device)).mean(dim=(2,3))
                ide = name.split(os.sep)[-1].split('.')[0]
                f.create_dataset(ide, data = output_features.cpu().numpy())
