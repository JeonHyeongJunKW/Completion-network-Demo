import torch
import torch.utils.data as data
from glob import glob
import cv2
from torchvision import transforms

    

class PlaceTwoDataset(data.Dataset):
  def __init__(self):
    super(PlaceTwoDataset.self).__init__()
    self.dataset = glob("")
    

  
  def __getitem__(self,index):
    #특정 인덱스의 사진과 
    if torch.is_tensor(index):
      index = index.tolist()
      
    image = cv2.imread(self.dataset[index])
    self.transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
      transforms.RandomResizedCrop((256,256),scale=(0.08,1),ratio=(0.5,1.5))]
    )
    t_image = self.transform(image) #이미지 자체도 랜덤크롭 (기존) [256, 384] / 
    #랜덤 크롭핑을 합니다.
    


    random_hole = t_image #임의 홈은 (기존)[96, 128]
    #빈공간을 

    return  t_image, 

  def __len__(self):
    return 10