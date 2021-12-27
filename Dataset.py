import torch
import torch.utils.data as data
from glob import glob
import cv2
from torchvision import transforms
import numpy as np
from Mask_Maker import *


class PlaceTwoDataset(data.Dataset):
  def __init__(self):
    self.dataset = glob("/media/jeon/T7/Place2/train_256_places365standard/data_256/**/**/*.jpg")
    print("dataset size : ",len(self.dataset))
    self.transform1 = transforms.Compose(
      [transforms.ToTensor(), #텐서변환을 잠시 끄자. 
      transforms.RandomResizedCrop((256,256),scale=(0.08,1),ratio=(0.5,1.5))]
    )
    self.transform2 = transforms.Compose(
      [transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
    )
  
  def __getitem__(self,index):
    
    mask = np.ones((256,256,1))
    random_param = np.random.rand(1,4)#시작 x,y / width, height
    random_param[0,0:2] = random_param[0,0:2]*128
    random_param[0,2:] = random_param[0,2:]*32+96
    random_param = random_param.astype(np.int64)

    mask_center_x = (random_param[0,0]+random_param[0,2]/2)
    mask_center_y = (random_param[0,1]+random_param[0,3]/2)

    mask[random_param[0,1]:random_param[0,1]+random_param[0,3], random_param[0,0]:random_param[0,0]+random_param[0,2]] = 0#구멍이 있는 이미지
    mask_rgb = np.ones((256,256,3))
    mask_rgb[random_param[0,1]:random_param[0,1]+random_param[0,3], random_param[0,0]:random_param[0,0]+random_param[0,2]] = 0#구멍이 있는 이미지
    #특정 인덱스의 사진과 
    
    if torch.is_tensor(index):
      index = index.tolist()
    image = cv2.imread(self.dataset[index])
    unmasked_image = self.transform1(image) #이미지 자체도 랜덤크롭 (기존) [256, 384] / 
    #랜덤 크롭핑을 합니다.

    
    masked_image = unmasked_image.permute(1,2,0).numpy().copy()
    masked_image[mask_rgb ==0] =0

    unmasked_image = self.transform2(unmasked_image).type(torch.float)
    masked_image =torch.from_numpy(masked_image).permute(2,0,1)
    masked_image = self.transform2(masked_image).type(torch.float)#기존에 넘파이로 만들어서 수정하였기때문에, 텐서로 바꾼후에 정규화한다. 
    mask = torch.from_numpy(mask).permute(2,0,1).type(torch.float)
    #빈공간을 
    return  unmasked_image, masked_image, mask, mask_center_x, mask_center_y

  def __len__(self):
    return len(self.dataset)

  
class PlaceTwoTestDataset(data.Dataset):
  def __init__(self):
    self.dataset = glob("/media/jeon/새 볼륨/place/test_256/*.jpg")
    print("Test dataset size : ",len(self.dataset))
    self.transform1 = transforms.Compose(
      [transforms.ToTensor(), #텐서변환을 잠시 끄자. 
      transforms.RandomResizedCrop((256,256),scale=(0.08,1),ratio=(0.5,1.5))]
    )
    self.transform2 = transforms.Compose(
      [transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
    )
  
  def __getitem__(self,index):
    
    mask = np.ones((256,256,1))
    random_param = np.random.rand(1,4)#시작 x,y / width, height
    random_param[0,0:2] = random_param[0,0:2]*128
    random_param[0,2:] = random_param[0,2:]*32+96
    random_param = random_param.astype(np.int64)

    mask_center_x = (random_param[0,0]+random_param[0,2]/2)
    mask_center_y = (random_param[0,1]+random_param[0,3]/2)

    mask[random_param[0,1]:random_param[0,1]+random_param[0,3], random_param[0,0]:random_param[0,0]+random_param[0,2]] = 0#구멍이 있는 이미지
    mask_rgb = np.ones((256,256,3))
    mask_rgb[random_param[0,1]:random_param[0,1]+random_param[0,3], random_param[0,0]:random_param[0,0]+random_param[0,2]] = 0#구멍이 있는 이미지
    #특정 인덱스의 사진과 
    
    if torch.is_tensor(index):
      index = index.tolist()
    image = cv2.imread(self.dataset[index])
    unmasked_image = self.transform1(image) #이미지 자체도 랜덤크롭 (기존) [256, 384] / 
    #랜덤 크롭핑을 합니다.

    
    masked_image = unmasked_image.permute(1,2,0).numpy().copy()
    masked_image[mask_rgb ==0] =0

    unmasked_image = self.transform2(unmasked_image).type(torch.float)
    masked_image =torch.from_numpy(masked_image).permute(2,0,1)
    masked_image = self.transform2(masked_image).type(torch.float)#기존에 넘파이로 만들어서 수정하였기때문에, 텐서로 바꾼후에 정규화한다. 
    mask = torch.from_numpy(mask).permute(2,0,1).type(torch.float)
    #빈공간을 
    return  unmasked_image, masked_image, mask, mask_center_x, mask_center_y

  def __len__(self):
    return len(self.dataset)