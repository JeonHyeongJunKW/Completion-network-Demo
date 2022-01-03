from torch.cuda import is_available
from Network import *
from Dataset import *
#데이터로더 만들고 
##데이터셋 만드는거 하고
dataset = PlaceTwoDataset()
dataset_test = PlaceTwoTestDataset()
#학습 전반적으로 중간중간에 표시및 바꿔주는거 진행

from torch.utils.data import DataLoader
from Draw2Writer import *
from tensorboardX import SummaryWriter
from Mask_Maker import * 
#외부적인 요소 
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter("runs/Train")


Tc = 500#90000 -> 140분의 1
Td = 100#10000 -> 100분의 1
Ttrain = 5000#500000 -> 100분의 1
Tsum = Tc+Td+Ttrain

#모델 및 데이터셋 선언
train_dataloader = DataLoader(dataset, batch_size=96,shuffle=True,num_workers=4)#1,434,892개의 이미지 
test_dataloader = DataLoader(dataset_test, batch_size=1,shuffle=True,num_workers=4)#1,434,892개의 이미지 
Generator = Completion_Network().to(device)
Des_Net = Discriminator(256,256,128,128).to(device)


##옵티마이저 및 손실함수 정의 
OptimizerG = torch.optim.Adam(Generator.parameters(),lr=0.001)
OptimizerD = torch.optim.Adam(Des_Net.parameters(),lr=0.001)

loss_fn_1 = nn.MSELoss(reduction='mean') 
loss_fn_2 = nn.BCELoss(reduction='mean')

#Generator의 출력이 1:1로 이미지가 같아지게하는 nn.Mse loss


iteration = 0
print("사용하는 디바이스", device)
for epoch in range(Tsum):
  for i , (unmasked_images, masked_images, masks, mask_center_x, mask_center_y) in enumerate(train_dataloader):
    unmasked_images= unmasked_images.to(device)
    masked_images = masked_images.to(device)
    masks = masks.to(device)
    
    MaskAndMaskedImage = torch.cat([masked_images,masks],dim=1).to(device)
    Gen_loss =0
    Des_Real_loss = 0
    Des_Fake_loss = 0
    full_Gen_loss = 0 
    if epoch < Tc:
      Generator.zero_grad()
      real_image = unmasked_images
      fake_image = Generator(MaskAndMaskedImage)
      synthesis_image =ImageSum(fake_image,masks,real_image)# 합성한 이미지 입니다.
      
      
      loss_mse = loss_fn_1(synthesis_image,real_image)
      loss_mse.backward()
      Gen_loss = loss_mse.item()
      OptimizerG.step()
    elif epoch < Tc+Td :
      Des_Net.zero_grad()
      real_image = unmasked_images
      random_center_x = (np.random.rand(real_image.size(0),1)*144+48).astype(np.int64)
      random_center_y = (np.random.rand(real_image.size(0),1)*144+48).astype(np.int64)
      real_local_image =GetLocalImage(real_image, random_center_x, random_center_y).to(device)#이미지에서 특정 중심의 128*128의 이미지를 얻어야함 

      y_real = Des_Net(real_image,real_local_image)

      real_label = torch.full((y_real.size(0),1), 1., dtype=torch.float,device=device)
      loss_bce_real = loss_fn_2(y_real,real_label)
      
      loss_bce_real.backward()#backward만 두개 해야한다.

      Des_Real_loss = loss_bce_real.item()
      
      with torch.no_grad():
        fake_image = Generator(MaskAndMaskedImage).to(device)
      #출력된 페이크 이미지에서 마스크 만큼 잘라주는게 필요 
      synthesis_image =ImageSum(fake_image,masks,real_image)# 합성한 이미지 입니다.
      fake_local_image =GetLocalImage(synthesis_image, mask_center_x.numpy().reshape(-1,1).astype(np.int64), mask_center_y.numpy().reshape(-1,1).astype(np.int64)).to(device)#이미지에서 특정 중심의 128*128의 이미지를 얻어야함 
      
      y_fake = Des_Net(synthesis_image,fake_local_image)
      fake_label = torch.full((y_fake.size(0),1), 0., dtype=torch.float,device=device)
      loss_bce_fake = loss_fn_2(y_fake,fake_label)
      loss_bce_fake.backward()#backward만 두개 해야한다.
      Des_Fake_loss = loss_bce_fake.item()
      OptimizerD.step()
        #descriminator 구분하기 
    else:
      Des_Net.zero_grad()
      real_image = unmasked_images
      random_center_x = (np.random.rand(real_image.size(0),1)*144+48).astype(np.int64)
      random_center_y = (np.random.rand(real_image.size(0),1)*144+48).astype(np.int64)
      real_local_image =GetLocalImage(real_image, random_center_x, random_center_y).to(device)#이미지에서 특정 중심의 128*128의 이미지를 얻어야함 

      y_real = Des_Net(real_image,real_local_image)

      real_label = torch.full((y_real.size(0),1), 1., dtype=torch.float,device=device)
      loss_bce_real = loss_fn_2(y_real,real_label)
      Des_Real_loss = loss_bce_real.item()
      loss_bce_real.backward()

      
      
      fake_image = Generator(MaskAndMaskedImage).to(device)
      synthesis_image =ImageSum(fake_image,masks,real_image)
      fake_local_image =GetLocalImage(synthesis_image, mask_center_x.numpy().reshape(-1,1).astype(np.int64), mask_center_y.numpy().reshape(-1,1).astype(np.int64)).to(device)#이미지에서 특정 중심의 128*128의 이미지를 얻어야함 
      y_fake = Des_Net(synthesis_image.detach(),fake_local_image.detach())
      fake_label = torch.full((y_fake.size(0),1), 0., dtype=torch.float,device=device)
      loss_bce_fake = loss_fn_2(y_fake,fake_label)
      loss_bce_fake.backward()
      Des_Fake_loss = loss_bce_fake.item()
      OptimizerD.step()

      Generator.zero_grad()
      y_fake = Des_Net(synthesis_image,fake_local_image)
      loss_G_fake = loss_fn_2(y_fake,real_label) +loss_fn_1(synthesis_image,real_image)*0.0004
      loss_G_fake.backward()
      full_Gen_loss = loss_G_fake.item()
      OptimizerG.step()


    if epoch %3 ==0 and i==0:
      
      with torch.no_grad():
        
        DrawMiddleTrainResult(writer, unmasked_images[0],masked_images[0],synthesis_image[0],epoch)
        rand_idx = np.random.randint(328500,size=1)[0]
        unmasked_images, masked_images, masks, mask_center_x, mask_center_y = dataset_test[rand_idx]
        unmasked_images= unmasked_images.view(1,3,256,256).to(device)
        masked_images = masked_images.view(1,3,256,256).to(device)
        masks = masks.view(1,1,256,256).to(device)
    
        MaskAndMaskedImage = torch.cat([masked_images,masks],dim=1).to(device)
        fake_image = Generator(MaskAndMaskedImage).to(device)
        synthesis_image =ImageSum(fake_image[0].unsqueeze(0),masks,unmasked_images)
        DrawMiddleTestResult(writer,unmasked_images[0],masked_images[0],synthesis_image[0],epoch)
        print("add new image")
    # print(Completed_image.shape)
  #모델 저장 
    if iteration%100 ==0:
        '''
        Gen_loss =0
        Des_Real_loss = 0
        Des_Fake_loss = 0
        full_Gen_loss = 0 
        '''
        writer.add_scalar("Generator loss", Gen_loss,iteration/100)
        writer.add_scalar("Des_Real loss", Des_Real_loss,iteration/100)
        writer.add_scalar("Des_Fake loss", Des_Fake_loss,iteration/100)
        writer.add_scalar("Full_Gen loss", full_Gen_loss,iteration/100)
        print("iter: ",iteration/100,"gen loss : ", Gen_loss," Des Real loss : ",Des_Real_loss, " Des_Fake loss : ",Des_Fake_loss," Full_Gen loss : ", full_Gen_loss)
    iteration +=1
  if epoch %10 ==0:
    torch.save(Generator.state_dict(),
                        "./model_weight/Generator/epoch_"+str(epoch)+"_weight.pth")
    torch.save(Des_Net.state_dict(),
                        "./model_weight/Discriminator/epoch_"+str(epoch)+"_weight.pth")
    print("현재 epoch : ", epoch)
writer.close()
