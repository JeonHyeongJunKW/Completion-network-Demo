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