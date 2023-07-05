import os
import pandas as pd
from torch.utils.data import Dataset
from math import ceil
import cv2 as cv
import torch
import numpy as np
import sklearn.neighbors as nn





class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, train=False, test=False, val=False, batch_size=64,shuffle=False,random_state=5):
        self.img_dir = img_dir
        self.random_state=random_state
        self.transform = transform
        self.target_transform = target_transform
        self.train_path_list=[]
        self.train_labels=[]
        self.test_path_list=[]
        self.test_labels=[]
        self.batch_size=batch_size
        self.train=train
        self.test=test
        self.val=val
        self.img_dir=img_dir
        self.shuffle=shuffle


        self.createiter()



    def __len__(self):
        return len(self.all_images)
    
    def createiter(self):
        for top, _, files in os.walk(self.img_dir):
            for j in files:
                self.train_path_list.append(os.path.join(top,j))
        self.imgs=pd.DataFrame.from_dict({'Address':self.train_path_list})
        if(self.test==True):
            self.imgs = self.imgs.tail(int(len(self.imgs)*0.002))
        else:
            self.imgs = self.imgs.head(int(len(self.imgs)*0.008))
        if(self.shuffle):
            self.imgs=self.imgs.sample(frac=1,random_state=self.random_state)
        if(self.val==True):
            self.imgs = self.imgs.tail(int(len(self.imgs)*0.2))
        elif(self.train==True):
            self.imgs = self.imgs.head(int(len(self.imgs)*0.8))
        self.all_images=[self.imgs.iloc[i*self.batch_size:min((i+1)*self.batch_size,len(self.imgs))] for i in range(ceil(len(self.imgs)/self.batch_size))]
        self.resetiter()

    def resetiter(self):
        self.iter=iter(self.all_images)


    def __getitem__(self):
        imageslabels=next(self.iter)
        img_path = list(imageslabels.values)
        images = []
        labels = []
        for i in range(len(img_path)):
            bgr = cv.imread(img_path[i][0])
            gray = cv.imread(img_path[i][0], 0)

            
            bgr = cv.resize(bgr, (256, 256), cv.INTER_CUBIC)
            gray = cv.resize(gray, (256, 256), cv.INTER_CUBIC)

        
            lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
            out_ab = self.target_transform(lab[:, :, 1:])
            out_l = self.target_transform(lab[:, :, 0])
            images.append(self.transform(out_l[0]))         
            labels.append(out_ab)
        return torch.stack(images),torch.Tensor(np.stack(labels, axis=0))

    


from torchvision import transforms

transforms_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])


transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),])



if __name__=="__main__":
    image=CustomImageDataset("../datasets/landscapes/", train=True, test=False, val=False, batch_size=2, shuffle=True, transform=transforms_train, target_transform=transforms_test)
    print(image.__getitem__()[1])
    print(image.__len__())