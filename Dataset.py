import torch
import torch.utils.data as data
from glob import glob
import cv2
from torchvision import transforms
import numpy as np

    

class PlaceTwoDataset(data.Dataset):
  def __init__(self):
    self.dataset = glob("/media/jeon/T7/Place2/train_256_places365standard/data_256/**/**/*.jpg")
    print("dataset size : ",len(self.dataset))
    self.transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
      transforms.RandomResizedCrop((256,256),scale=(0.08,1),ratio=(0.5,1.5))]
    )
  
  def __getitem__(self,index):
    #특정 인덱스의 사진과 
    if torch.is_tensor(index):
      index = index.tolist()
    image = cv2.imread(self.dataset[index])
    t_image = self.transform(image) #이미지 자체도 랜덤크롭 (기존) [256, 384] / 
    #랜덤 크롭핑을 합니다.
    
    hole_image = np.ones((256,256))
    random_param = np.random.rand(1,4)#시작 x,y / width, height
    random_param[0,0:2] = random_param[0,0:2]*128
    random_param[0,2:] = random_param[0,2:]*32+96
    random_param = random_param.astype(np.int64)
    hole_image[random_param[0,1]:random_param[0,1]+random_param[0,3],random_param[0,0]:random_param[0,0]+random_param[0,2]]#구멍이 있는 이미지
    hole_image = torch.from_numpy(hole_image)
    #빈공간을 

    return  t_image, hole_image

  def __len__(self):
    return len(self.dataset)