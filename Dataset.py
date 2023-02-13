import torch
import torch.utils.data as data
from glob import glob
import cv2
from torchvision import transforms
import numpy as np
from Mask_Maker import *


class PlaceTwoDataset(data.Dataset):
  # train 데이터 셋
  def __init__(self):
    self.dataset = glob("/media/jeon/새 볼륨/place/data_256/**/**/*.jpg")# glob 모듈을 사용하여 해당 폴더내에서 이미지들의 파일 이름을 얻어옵니다.
    print("dataset size : ",len(self.dataset)) # 데이터 크기를 표현한다.

    # 이미지의 일부 크롭 및 노멀라이즈를 합니다.
    self.transform1 = transforms.Compose(
      [transforms.ToTensor(), # 이미지를 pytorch의 tensor로 변경합니다. 이미지 스케일을 0~255 -> 0~1로 변경하는 기능도 있습니다.
      transforms.RandomResizedCrop((256,256),scale=(0.08,1),ratio=(0.5,1.5))] # 원본이미지 리사이즈, 크롭 및 BGR 이미지를 노멀라이즈합니다.
    )
    # 이미지 노멀라이즈를 합니다.
    self.transform2 = transforms.Compose(
      [transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))] # BGR 이미지를 노멀라이즈합니다. scale : 0~1 -> -1 ~ 1
    )
  
  def __getitem__(self, index):

    # 1. 마스크를 만듭니다.
    mask = np.zeros((256,256,1)) # 기본 마스크 생성, 0으로 채워진 이미지

    # 1-1. 마스크에서 가릴 부분을 정하고, 해당 부분을 1로 채웁니다. (이 부분은 수정할 수 있습니다.)
    random_param = np.random.rand(1,4)#[x, y, width, height]를 0~1 사이의 범위로 저장합니다.
    random_param[0,0:2] = random_param[0,0:2]*128# x, y를 0~128 사이로 스케일을 조정합니다.
    random_param[0,2:] = random_param[0,2:]*32+96# width, height를 96 ~ 128 사이로 조정합니다.
    random_param = random_param.astype(np.int64)# 픽셀단위로 마스크를 만들기 위해서, 정수형으로 수정합니다.
    mask_center_x = (random_param[0,0]+random_param[0,2]/2)# 마스크 중심 x좌표 =(마스크 x좌표 + 마스크_width/2 )
    mask_center_y = (random_param[0,1]+random_param[0,3]/2)# 마스크 중심 y좌표 =(마스크 y좌표 + 마스크_height/2 )
    mask[random_param[0, 1]:random_param[0, 1] + random_param[0, 3],
          random_param[0, 0]:random_param[0, 0] + random_param[0, 2]] = 1 # 가릴부분을 1로 채운다.

    # 2. 원본 이미지(BGR)에 적용할 3채널 마스크입니다.
    mask_rgb = np.zeros((256,256,3))
    mask_rgb[random_param[0, 1]:random_param[0, 1]+random_param[0, 3],
              random_param[0, 0]:random_param[0, 0]+random_param[0, 2]] = 1 # 가릴부분을 1로 만듭니다.

    # ※이거는 무시하거나 지워도 될듯합니다.
    if torch.is_tensor(index):
      index = index.tolist()

    # 3. 데이터셋(이미지 이름)에서 데이터를 읽어오고, 전처리합니다.
    image = cv2.imread(self.dataset[index])# glob 했던 파일이름으로 원본이미지를 읽어옵니다.
    unmasked_image = self.transform1(image)# 이미지 전처리를 합니다. (입력) [256, 384, 3] -> (출력) [3, 256, 256] 스케일 0~1

    # 4. 잠시 tensor에서 numpy로 수정한 후에 마스크 연산을 적용합니다.
    masked_image = unmasked_image.permute(1,2,0).numpy().copy() # 잠시 numpy로 수정
    masked_image[mask_rgb ==1] = 1. # 마스크된 부분을 1로 수정합니다.

    # 5. 마스크된 이미지와 원본 이미지에 노멀라이즈 연산을 적용합니다.
    unmasked_image = self.transform2(unmasked_image).type(torch.float)# 원본 이미지를 노멀라이즈합니다. scale : 0~1 -> -1 ~ 1
    masked_image = torch.from_numpy(masked_image).permute(2,0,1)# 원본 이미지를 torch자료형으로 변환합니다.
    masked_image = self.transform2(masked_image).type(torch.float)# 마스크된 이미지를 노멀라이즈합니다. scale : 0~1 -> -1 ~ 1

    # 6. 모델에 넣을 마스크를 torch형태로 만듭니다. scale : 0~1
    mask = torch.from_numpy(mask).permute(2,0,1).type(torch.float)

    # 7. 원본이미지, 마스크된 이미지, 마스크, 마스크 x좌표, 마스크 y좌표를 반환합니다.
    return unmasked_image, masked_image, mask, mask_center_x, mask_center_y

  def __len__(self):
    # 데이터셋의 갯수를 임의로 조정할 수 있습니다. 이부분은 본인이 수정해야합니다.
    return 10000

  
class PlaceTwoTestDataset(data.Dataset):
  def __init__(self):
    # 주석이 없는 이유 : 위쪽의 Train 데이터셋과 다른 점은 데이터셋 경로위치와 갯수입니다. 사실 이렇게 중복되게 만들필요는 없습니다.
    self.dataset = glob("/media/jeon/새 볼륨/place/test_256/*.jpg")
    print("Test dataset size : ",len(self.dataset))
    self.transform1 = transforms.Compose(
      [transforms.ToTensor(),
      transforms.RandomResizedCrop((256,256),scale=(0.08,1),ratio=(0.5,1.5))]
    )
    self.transform2 = transforms.Compose(
      [transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
    )
  
  def __getitem__(self,index):
    
    mask = np.zeros((256,256,1))
    random_param = np.random.rand(1,4)
    random_param[0,0:2] = random_param[0,0:2]*128
    random_param[0,2:] = random_param[0,2:]*32+96
    random_param = random_param.astype(np.int64)

    mask_center_x = (random_param[0,0]+random_param[0,2]/2)
    mask_center_y = (random_param[0,1]+random_param[0,3]/2)

    mask[random_param[0,1]:random_param[0,1]+random_param[0,3], random_param[0,0]:random_param[0,0]+random_param[0,2]] = 1
    mask_rgb = np.zeros((256,256,3))
    mask_rgb[random_param[0,1]:random_param[0,1]+random_param[0,3], random_param[0,0]:random_param[0,0]+random_param[0,2]] = 1
    
    if torch.is_tensor(index):
      index = index.tolist()
    image = cv2.imread(self.dataset[index])
    unmasked_image = self.transform1(image)
    
    masked_image = unmasked_image.permute(1,2,0).numpy().copy()
    masked_image[mask_rgb ==1] =1

    unmasked_image = self.transform2(unmasked_image).type(torch.float)
    masked_image =torch.from_numpy(masked_image).permute(2,0,1)
    masked_image = self.transform2(masked_image).type(torch.float)
    mask = torch.from_numpy(mask).permute(2,0,1).type(torch.float)
    return  unmasked_image, masked_image, mask, mask_center_x, mask_center_y

  def __len__(self):
    return 100