import torch
from torch import functional
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear

class Discriminator(nn.Module):
  def __init__(self,image_width, image_height, local_image_width, local_image_height):
      super().__init__()
      self.image_width = image_width
      self.image_height = image_height
      self.flattenOutput = int(self.Input2Output(self.image_width,self.image_width,"Global"))
      self.local_image_width = local_image_width
      self.local_image_height = local_image_height
      self.local_flattenOutput = int(self.Input2Output(self.local_image_width,self.local_image_width,"local"))
    
      self.Local_Discriminator = nn.Sequential(
        nn.Conv2d(3,64,5,2,padding=2),#(Hin +2*padding -4-1)
        nn.Conv2d(64,128,5,2,padding=2),
        nn.Conv2d(128,256,5,2,padding=2),
        nn.Conv2d(256,512,5,2,padding=2),
        nn.Conv2d(512,512,5,2,padding=2),
        nn.Flatten(),
        nn.Linear(self.local_flattenOutput, out_features=1024)
      )
      
      self.Global_Discriminator = nn.Sequential(
        nn.Conv2d(3,64,5,2,padding=2),#
        nn.Conv2d(64,128,5,2,padding=2),
        nn.Conv2d(128,256,5,2,padding=2),
        nn.Conv2d(256,512,5,2,padding=2),
        nn.Conv2d(512,512,5,2,padding=2),
        nn.Conv2d(512,512,5,2,padding=2),
        nn.Flatten(),
        nn.Linear(self.flattenOutput,out_features=1024)
      )
      self.Concatentation = nn.Sequential(
        nn.Linear(2048,1),
        nn.Sigmoid()
      )


  def Input2Output(self,imageWidth,imageHeight,OutputType):
    return 512*16
  
  def forward(self, full_image,local_image):
    full_dis = self.Global_Discriminator(full_image)
    local_dis = self.Local_Discriminator(local_image)
    sum_dis = torch.cat([full_dis,local_dis],dim=1)#벡터만 다르므로 
    output = self.Concatentation(sum_dis)
    return output

class Completion_Network(nn.Module):

  def __init__(self):
      super().__init__()
      self.conv_model = nn.Sequential(
        nn.Conv2d(4,64,5,1,dilation=1,padding=2),# 
        nn.ReLU(),
        nn.Conv2d(64,128,3,2,dilation=1,padding=1),#절반 감소 패딩합은 -0.5 +1 
        nn.ReLU(),
        nn.Conv2d(128,128,3,1,dilation=1,padding=1),#
        nn.ReLU(),
        nn.Conv2d(128,256,3,2,dilation=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=1,padding=1),
        nn.ReLU(),
      )
      self.dilated_conv_model = nn.Sequential(
        nn.Conv2d(256,256,3,1,dilation=2,padding=2),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=4,padding=4),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=8,padding=8),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=16,padding=16),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(256,256,3,1,dilation=1,padding=1),
        nn.ReLU(),
      )
      self.deconv_model = nn.Sequential(
        nn.ConvTranspose2d(256,128,4,2,dilation=1,padding=1),# (Hin -1 )*2 -2*1 + 1*(4-1) +1
        nn.ReLU(),
        nn.Conv2d(128,128,3,1,dilation=1,padding=1),# (Hin -1)*2 -2*2 + 1*(3-1) +1
        nn.ReLU(),


        nn.ConvTranspose2d(128,64,4,2,dilation=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(64,32,3,1,dilation=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(32,3,3,1,dilation=1,padding=1),
        nn.Tanh()
      )

  def forward(self,X):
    input1 = self.conv_model(X)
    input2 = self.dilated_conv_model(input1)
    output = self.deconv_model(input2)
    return output


