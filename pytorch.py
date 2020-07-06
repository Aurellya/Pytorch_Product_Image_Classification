#==============================================================#
#===========================NOTES==============================#
#==============================================================#

# find . -name '.DS_Store' -type f -delete.
# to delete .DS file in image folder

#=================================================================#
#=========================IMPORT LIBRARY==========================#
#=================================================================#

import os
import numpy as np
import pandas as pd
from PIL import Image,ImageFile
from tqdm.notebook import tqdm

import torchvision.models as models
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

#=================================================================#
#==========LIST THE AMOUNT OF IMAGE FILE IN EACH FOLDER===========#
#=================================================================#

ImageFile.LOAD_TRUNCATED_IMAGES = True

# List the amount of images in each category
path = "shopee-product-detection-dataset/train/train/"
classes = ["%.2d" % i for i in range(len(os.listdir(path)))]
for i in classes:
    if(i == '42'):
      break
    file = sorted(os.listdir(path+i))
    print("Number of images for Class {}:".format(i),len(file))


#=================================================================#
#============CLASSES AND FUNCTIONS FOR DATA TRAINING==============#
#=================================================================#

class Dataset(Dataset):
    def __init__(self,directory,transform,train=True):
        self.transform = transform                      
        self.img_ls = list()                            

        classes = ["%.2d" % i for i in range(len(os.listdir(directory)))]           
        for class_ in classes:
            if(class_=='42'): # to prevent .DS Store
              break
            path = os.path.join(directory,class_)
            ls = sorted(os.listdir(path))            
            if train:
                for img_name in ls[:-50]:                                           
                    self.img_ls.append((os.path.join(path,img_name),int(class_)))     
            else:
                for img_name in ls[-50:]:                                           
                    self.img_ls.append((os.path.join(path,img_name),int(class_)))
        
    def __getitem__(self, index):               
        name, label = self.img_ls[index]                 
        img = Image.open(name).convert('RGB')           
        img.load()
        img = self.transform(img)                   
        return {"image": img, "label": label}
    
    def __len__(self):                                  
        return len(self.img_ls)


def train_epoch(model,  trainloader,  criterion, device, optimizer):
    model.train()
    losses = list()
    i=0
    for batch_idx, data in enumerate(tqdm(trainloader)):       
        i+=1
        inputs=data['image'].to(device)                 
        labels=data['label'].to(device)                 
        optimizer.zero_grad()
        outputs = model(inputs)                        
        loss = criterion(outputs, labels)              
        loss.backward()                                 
        optimizer.step()                               
        losses.append(loss.item())
    return sum(losses)/i


def evaluate(model, dataloader, device):
    model.eval()                                                    
    total=0                                                        
    correct = 0                                                    
    with torch.no_grad():
      for ctr, data in enumerate(dataloader):
          inputs = data['image'].to(device)                        
          outputs = model(inputs)                                   
          labels = data['label']                                   
          labels = labels.float()
          cpuout= outputs.to('cpu')
          total += len(labels)                                      

          pred = cpuout.argmax(dim=1, keepdim=True)                 
          correct += pred.eq(labels.view_as(pred)).sum().item()     

      accuracy = correct / total                                    
    return accuracy

def train_modelcv(dataloader_cvtrain, dataloader_cvtest,  model,  criterion, optimizer, scheduler, num_epochs, device):
  for epoch in range(num_epochs):                                                         
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train(True)                                                                           
    train_loss=train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )   

    model.train(False)
    measure = evaluate(model, dataloader_cvtest, device)                                    

    print('Top 1 Accuracy:', measure, "\n")                                                 
  return None


#=================================================================#
#=============CLASSES AND FUNCTIONS FOR DATA TESTING==============#
#=================================================================#

class TestDataset(Dataset):
    def __init__(self,directory,transform):
        self.transform = transform
        self.img_ls = list()

        ls = sorted(os.listdir(directory))
        for img_name in ls:
            self.img_ls.append((os.path.join(directory,img_name)))
        
    def __getitem__(self, idx):
        name = self.img_ls[idx]
        img = Image.open(name).convert('RGB')
        img.load()
        img = self.transform(img)
        return img,name[12:]
    
    def __len__(self):
        return len(self.img_ls)

def inference(model, dataloader, device):
    path_ls = list()
    pred_ls = list()
    model.eval()
    with torch.no_grad():
      for ctr, data in enumerate(dataloader):
          inputs,path = data
          inputs = inputs.to(device)
          outputs = model(inputs)
          cpuout= outputs.to('cpu')
          pred = cpuout.argmax(dim=1).numpy()
          path_ls += list(path)
          pred_ls += list(pred)
    df = pd.DataFrame({'filename':path_ls,'category':pred_ls})
    return df


#=================================================================#
#===================DEFINE MODEL AND TRAIN========================#
#=================================================================#

BATCHSIZE = 64                                                            
sp = 0.01     

data_transform = transforms.Compose([transforms.Resize((224,224)),      
                                     transforms.RandomCrop(size=(224,224),padding=(10,10)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomRotation(degrees=15,fill=0),
                                     transforms.ToTensor(),             
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])   

path = "shopee-product-detection-dataset/train/train/"                        
dataset_train = Dataset(path, data_transform,train=True)          
dataset_valid = Dataset(path, data_transform,train=False)

loadertr=torch.utils.data.DataLoader(dataset_train, batch_size=BATCHSIZE, shuffle=True)       
loaderval=torch.utils.data.DataLoader(dataset_valid, batch_size=BATCHSIZE, shuffle=True)     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
criterion = nn.CrossEntropyLoss()                                       

model = models.resnet18(pretrained=True)                                
model.fc = nn.Linear(512, 42)                                           
model.to(device)                                                       

optimizer = torch.optim.SGD(model.parameters(), lr=sp, momentum=0.9, weight_decay=5e-4)
epochs=8                                                          

train_modelcv(dataloader_cvtrain = loadertr,                            
              dataloader_cvtest = loaderval,  
              model = model,  
              criterion = criterion, 
              optimizer = optimizer,
              scheduler = None,
              num_epochs = epochs,
              device = device)


#=================================================================#
#=============================TESTING=============================#
#=================================================================#

test_dataset = TestDataset("shopee-product-detection-dataset/test/test/", data_transform)
loaderte=torch.utils.data.DataLoader(test_dataset,batch_size=256,shuffle=False)
df = inference(model,loaderte,device)        
df.to_csv("output.csv",index=False)      

df.head()






