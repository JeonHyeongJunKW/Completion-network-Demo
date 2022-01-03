import numpy as np
import torch 
def GetLocalImage(images, local_x, local_y):
  #실제 이미지가 
  diff_x = local_x-65
  local_x[diff_x <0] =local_x[diff_x <0] -diff_x[diff_x <0]
  diff_x = local_x-191#256- 64 =192
  local_x[diff_x >0] =local_x[diff_x >0] -diff_x[diff_x >0]
  diff_y = local_y-65
  local_y[diff_y <0] =local_y[diff_y <0] -diff_y[diff_y <0]
  diff_y = local_y-191
  local_y[diff_y >0] =local_y[diff_y >0] -diff_y[diff_y >0]
  cropped_image = images[0,:,local_y[0,0]-64:local_y[0,0]+64,local_x[0,0]-64:local_x[0,0]+64]
  cropped_image = torch.unsqueeze(cropped_image,0)
  for i in range(1,images.size(0)):
    temp_image = images[i,:,local_y[i,0]-64:local_y[i,0]+64,local_x[i,0]-64:local_x[i,0]+64]
    temp_image = torch.unsqueeze(temp_image,0)
    cropped_image = torch.cat([cropped_image,temp_image],dim=0)
  return cropped_image


def ImageSum(InputImage, Mask, RealImage):
  '''
  256*256 이미지 
  >>> image[test<3] =0
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  IndexError: The shape of the mask [2, 2, 2] at index 1 does not match the shape of the indexed tensor [2, 3, 2, 2] at index 1
  '''
  triple_mask = Mask.expand(-1,3,-1,-1)# batch 1*width height로 늘리고, 차원을 3차원으로 늘림 -> 이미 배치로 나오면서 차원이 늘어나있음
  ReturnImage = InputImage.clone()
  ReturnImage[triple_mask ==0] = 0#마스킹연산
  RealImage_copy =RealImage.clone()
  RealImage_copy[triple_mask ==1] = 0#반대로 마스킹
  ReturnImage = torch.add(ReturnImage,RealImage_copy)# 더하기 연산
  return ReturnImage
