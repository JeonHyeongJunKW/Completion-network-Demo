from Network import *
from Dataset import *
#데이터로더 만들고 
##데이터셋 만드는거 하고
dataset = PlaceTwoDataset()
#학습 전반적으로 중간중간에 표시및 바꿔주는거 진행

from torch.utils.data import DataLoader


train_dataloader = DataLoader(dataset, batch_size=96,shuffle=True)#1,434,892개의 이미지 
Generator = Completion_Network()
Descriminator = Discriminator(256,256,128,128)
Tc = 900#90000 -> 100분의 1
Td = 100#10000 -> 100분의 1
Ttrain = 5000#500000 -> 100분의 1
Tsum = Tc+Td+Ttrain
for epoch in range(Tsum):
  for i , (t_images, hole_images) in enumerate(train_dataloader):
    '''
    print(i)
    print(t_images.shape)
    print(hole_images.shape)
    0
    torch.Size([96, 3, 256, 256])
    torch.Size([96, 256, 256])
    '''

    exit()