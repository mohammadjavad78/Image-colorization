import os
import torch
from torchvision import transforms
import torch.nn as nn
from dataloaders import  DataLoader
import matplotlib.pyplot as plt
from nets.model import ModelCNN
from utlis.Averagemeter import AverageMeter
import numpy as np
import cv2 as cv





def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    outputchanged=output.reshape(output.shape[0]*output.shape[1],output.shape[2])
    targetchanged=target.reshape(target.shape[0]*target.shape[1],target.shape[2])
    # print(outputchanged.shape)
    # print(targetchanged.shape)
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = targetchanged.size(0)
        # import numpy as np
        pred = outputchanged.argmax(axis=1)
        pred2 = targetchanged.argmax(axis=1)
        # print(pred)
        # pred = pred.t()
        # print(pred.shape)
        # print(targetchanged.shape)
        correct = pred.eq(pred2)
        # print(correct.shape)
        res = []
        # for k in topk:
        #     # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size).item())
        return res









def test(
    train_loader,
    val_loader,
    test_loader,
    model,
    model_name,
    device,
    ckpt_path,
    load_saved_model
):
    
    fig,axes= plt.subplots(2,3,figsize=(13,9))
    # fig.set_label("Some example of our result")

    model = model.to(device)



    if os.path.exists(f"{ckpt_path}/ckpt.ckpt"):
        if load_saved_model:
            model, _ = load_model(
                ckpt_path=f"{ckpt_path}/ckpt.ckpt", model=model
            )
    

    model.eval()
    mode = "train"
    
    
    (images, labels)=train_loader.__getitem__()
    (images, labels)=train_loader.__getitem__()
    # print(images.shape)
    labels=labels.reshape((labels.shape[0],labels.shape[1],256,256))
    labels=torch.from_numpy(np.concatenate([images,labels],axis=1))
    


    images = images.to(device)
    labels_pred = model(images).cpu()
    images = images.cpu()
    labels_pred = labels_pred.detach().numpy()
    labels_pred=labels_pred.reshape((labels_pred.shape[0],labels_pred.shape[1],256,256))
    # print(images.shape)
    # print(labels_pred.shape)
    labels_pred=torch.from_numpy(np.concatenate([images,labels_pred],axis=1))



    i=0
    dd=labels[0]
    dd[0]*=100
    dd[1:]*=254
    dd[1:]-=127
    dd=torch.permute(dd,(1,2,0)).numpy().astype(np.float32)
    dd=cv.cvtColor(dd, cv.COLOR_LAB2RGB)
    dd=torch.permute(torch.from_numpy(dd),(1,0,2)).numpy()
    dd = (dd - dd.min()) / (dd.max() - dd.min())
    dd=np.rot90(dd)
    dd=np.rot90(dd)
    dd=np.rot90(dd)
    dd=np.fliplr(dd)
    # axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
    axes[i][0].set_xticks([])
    axes[i][0].set_yticks([])
    axes[1][0].set_xlabel('train')
    axes[i][0].imshow(dd,aspect='auto')

    i=1
    dd=labels_pred[0]
    dd[0]*=100
    dd[1:]*=254
    dd[1:]-=127
    # print(dd)
    dd=torch.permute(dd,(1,2,0)).numpy().astype(np.float32)
    dd=cv.cvtColor(dd, cv.COLOR_LAB2RGB)
    dd=torch.permute(torch.from_numpy(dd),(1,0,2)).numpy()
    # dd = (dd - dd.min()) / (dd.max() - dd.min())
    dd=np.rot90(dd)
    dd=np.rot90(dd)
    dd=np.rot90(dd)
    dd=np.fliplr(dd)
    # axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
    axes[i][0].set_xticks([])
    axes[i][0].set_yticks([])
    axes[1][0].set_xlabel('train')
    axes[i][0].imshow(dd,aspect='auto')

        


    model.eval()
    mode = "val"
    with torch.no_grad():
    
        (images, labels)=val_loader.__getitem__()
        (images, labels)=val_loader.__getitem__()
        (images, labels)=val_loader.__getitem__()
            
        # print(images.shape)
        labels=labels.reshape((labels.shape[0],labels.shape[1],256,256))
        labels=torch.from_numpy(np.concatenate([images,labels],axis=1))
        


        images = images.to(device)
        labels_pred = model(images).cpu()
        images = images.cpu()
        labels_pred = labels_pred.detach().numpy()
        labels_pred=labels_pred.reshape((labels_pred.shape[0],labels_pred.shape[1],256,256))
        # print(images.shape)
        # print(labels_pred.shape)
        labels_pred=torch.from_numpy(np.concatenate([images,labels_pred],axis=1))



        i=0
        dd=labels[0]
        dd[0]*=100
        dd[1:]*=254
        dd[1:]-=127
        dd=torch.permute(dd,(1,2,0)).numpy().astype(np.float32)
        dd=cv.cvtColor(dd, cv.COLOR_LAB2RGB)
        dd=torch.permute(torch.from_numpy(dd),(1,0,2)).numpy()
        dd = (dd - dd.min()) / (dd.max() - dd.min())
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.fliplr(dd)
        # axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])
        axes[1][1].set_xlabel('val')
        axes[i][1].imshow(dd,aspect='auto')

        i=1
        dd=labels_pred[0]
        dd[0]*=100
        dd[1:]*=254
        dd[1:]-=127
        # print(dd[1:])
        dd=torch.permute(dd,(1,2,0)).numpy().astype(np.float32)
        dd=cv.cvtColor(dd, cv.COLOR_LAB2RGB)
        dd=torch.permute(torch.from_numpy(dd),(1,0,2)).numpy()
        dd = (dd - dd.min()) / (dd.max() - dd.min())
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.fliplr(dd)
        # axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
        axes[i][1].set_xticks([])
        axes[i][1].set_yticks([])
        axes[1][1].set_xlabel('val')
        axes[i][1].imshow(dd,aspect='auto')



    model.eval()
    mode = "test"
    with torch.no_grad():
        (images, labels)=test_loader.__getitem__()
        (images, labels)=test_loader.__getitem__()
        (images, labels)=test_loader.__getitem__()
        labels=labels.reshape((labels.shape[0],labels.shape[1],256,256))
        labels=torch.from_numpy(np.concatenate([images,labels],axis=1))
        


        images = images.to(device)
        labels_pred = model(images).cpu()
        images = images.cpu()
        labels_pred = labels_pred.detach().numpy()
        labels_pred=labels_pred.reshape((labels_pred.shape[0],labels_pred.shape[1],256,256))
        # print(images.shape)
        # print(labels_pred.shape)
        labels_pred=torch.from_numpy(np.concatenate([images,labels_pred],axis=1))



        i=0
        dd=labels[0]
        dd[0]*=100
        dd[1:]*=254
        dd[1:]-=127
        dd=torch.permute(dd,(1,2,0)).numpy().astype(np.float32)
        dd=cv.cvtColor(dd, cv.COLOR_LAB2RGB)
        dd=torch.permute(torch.from_numpy(dd),(1,0,2)).numpy()
        dd = (dd - dd.min()) / (dd.max() - dd.min())
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.fliplr(dd)
        # axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
        axes[i][2].set_xticks([])
        axes[i][2].set_yticks([])
        axes[1][2].set_xlabel('test')
        axes[i][2].imshow(dd,aspect='auto')

        i=1
        dd=labels_pred[0]
        dd[0]*=100
        dd[1:]*=254
        dd[1:]-=127
        dd=torch.permute(dd,(1,2,0)).numpy().astype(np.float32)
        dd=cv.cvtColor(dd, cv.COLOR_LAB2RGB)
        dd=torch.permute(torch.from_numpy(dd),(1,0,2)).numpy()
        dd = (dd - dd.min()) / (dd.max() - dd.min())
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.fliplr(dd)
        # axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
        axes[i][2].set_xticks([])
        axes[i][2].set_yticks([])
        axes[1][2].set_xlabel('test')
        axes[i][2].imshow(dd,aspect='auto')


    fig.suptitle(f'Some Example of our result')
    plt.show()
    # fig.savefig(f'{model_name}.png')
        
    # return model



from utlis import Read_yaml




yml=Read_yaml.Getyaml()



batch_size = yml['batch_size']
epochs = yml['num_epochs']
learning_rate = yml['learning_rate']
gamma=yml['gamma']
step_size=yml['step_size']
ckpt_save_freq = yml['ckpt_save_freq']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
custom_model = ModelCNN()

DIR = yml['dataset']

transforms_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

transforms_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()])

train_loader = DataLoader.CustomImageDataset(DIR,transforms_train,transforms_test,train=True,batch_size=batch_size,shuffle=True)

val_loader = DataLoader.CustomImageDataset(DIR,transforms_val,transforms_test,train=True,val=True,batch_size=batch_size,shuffle=True)

test_loader = DataLoader.CustomImageDataset(DIR,transforms_test,transforms_test,test=True,batch_size=batch_size,shuffle=True)

trainer = test(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    model = custom_model,
    model_name="model_name",
    device=device,
    ckpt_path=yml['ckpt_path'],
    load_saved_model=True
)

