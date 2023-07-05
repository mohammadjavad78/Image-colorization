import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from dataloaders import  DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from tqdm import tqdm
from nets.model import ModelCNN
from utlis.Averagemeter import AverageMeter
from utlis import Read_yaml





def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer






def test(
    train_loader,
    val_loader,
    test_loader,
    model,
    model_name,
    device,
    load_saved_model,
    ckpt_path,
    report_path,
):





    model = model.to(device)

    # loss function
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(f"{ckpt_path}/ckpt.ckpt"):
        if load_saved_model:
            model, optimizer = load_model(
                ckpt_path=f"{ckpt_path}/ckpt.ckpt", model=model, optimizer=optimizer
            )
    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_test_loss_till_current_batch",])

    loss_avg_train = AverageMeter()
    loss_avg_val = AverageMeter()
    loss_avg_test = AverageMeter()

    model.eval()
    mode = "train"
    
    
    loop_train = tqdm(range(train_loader.__len__()),
        total=train_loader.__len__(),
        desc="train",
        position=0,
        leave=True)
    for batch_idx in loop_train:

        (images, labels)=train_loader.__getitem__()

        images = images.to(device)
        labels = labels.to(device)
        

        labels_pred = model(images)
        loss = criterion(labels_pred, labels)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        loss_avg_train.update(loss.item(), images.size(0))

        new_row = pd.DataFrame(
            {"model_name": model_name,
                "mode": mode,
                "image_type":"original",
                "learning_rate":optimizer.param_groups[0]["lr"],
                "batch_size": images.size(0),
                "batch_index": batch_idx,
                "loss_batch": loss.detach().item(),
                "avg_train_loss_till_current_batch":loss_avg_train.avg,
                "avg_val_loss_till_current_batch":None,
                "avg_test_loss_till_current_batch":None,},index=[0])

        
        report.loc[len(report)] = new_row.values[0]
        
        loop_train.set_description(f"Train - iteration")
        loop_train.set_postfix(
            loss_batch="{:.4f}".format(loss.detach().item()),
            avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
            refresh=True,
        )

    model.eval()
    mode = "val"
    with torch.no_grad():
        loop_val = tqdm(range(val_loader.__len__()),
            total=val_loader.__len__(),
            desc="val",
            position=0,
            leave=True,
        )
        for batch_idx in loop_val:
            (images, labels)=val_loader.__getitem__()
            optimizer.zero_grad()
            images = images.to(device).float()
            labels = labels.to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            loss_avg_val.update(loss.item(), images.size(0))
            new_row = pd.DataFrame(
                {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": images.size(0),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_val_loss_till_current_batch":loss_avg_val.avg,
                    "avg_test_loss_till_current_batch":None,
                    },index=[0],)
            
            report.loc[len(report)] = new_row.values[0]
            loop_val.set_description(f"val - iteration")
            loop_val.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                refresh=True,
            )
            


    model.eval()
    mode = "test"
    with torch.no_grad():
        loop_test = tqdm(range(test_loader.__len__()),
            total=test_loader.__len__(),
            desc="test",
            position=0,
            leave=True,
        )
        for batch_idx in loop_test:
            (images, labels)=test_loader.__getitem__()
            optimizer.zero_grad()
            images = images.to(device).float()
            labels = labels.to(device)
            labels_pred = model(images)
            loss = criterion(labels_pred, labels)
            loss_avg_test.update(loss.item(), images.size(0))
            new_row = pd.DataFrame(
                {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": images.size(0),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_val_loss_till_current_batch":None,
                    "avg_test_loss_till_current_batch":loss_avg_test.avg,
                    },index=[0],)
            
            report.loc[len(report)] = new_row.values[0]
            loop_test.set_description(f"test - iteration")
            loop_test.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_test_loss_till_current_batch="{:.4f}".format(loss_avg_test.avg),
                refresh=True,
            )
    train_loader.resetiter()
    val_loader.resetiter()
    test_loader.resetiter()
        
    report.to_csv(f"{report_path}/{model_name}_report_test.csv")
    return model, optimizer, report


from utlis import Read_yaml

yml=Read_yaml.Getyaml()


model_name=yml['model_name']
# print()


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
    model_name=model_name,
    device=device,
    load_saved_model=True,
    ckpt_path=yml['ckpt_path'],
    report_path=yml['report_path'],
)
