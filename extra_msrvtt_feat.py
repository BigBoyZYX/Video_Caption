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


print(pretrainedmodels.model_names)
model_name = 'resnet101'#'inceptionresnetv2' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model = model.to(device)
model.eval()
print(pretrainedmodels.pretrained_settings[model_name])

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

dset_path = os.path.join(os.path.join(root,'Datasets'),'MSR-VTT')
msrvtt_trnval_path = '/home/meihaiwei/HDD/code_bak/video-caption.pytorch_raw/data/TrainValVideo'
msrvtt_test_path = '/home/meihaiwei/HDD/code_bak/video-caption.pytorch_raw/data/TestVideo'

msrvtt_trnval_name_list = glob.glob(msrvtt_trnval_path+os.sep+'*')
msrvtt_test_name_list = glob.glob(msrvtt_test_path+os.sep+'*')
msrvtt_name_list = msrvtt_trnval_name_list + msrvtt_test_name_list

save_file = os.path.join(save_path,'MSRVTT_inceptionresnetv2.hdf5')


with torch.no_grad():
    with h5py.File(save_file, 'w') as f:
        for name in tqdm(msrvtt_name_list):
            tensor,_,_ = video2tensor(name,28)
            output_features = model.features(tensor.to(device)).mean(dim=(2,3))
            ide = name.split(os.sep)[-1].split('.')[0]
            f.create_dataset(ide, data = output_features.cpu().numpy())



