from typing import Generator
import torch
from Network import *
from Dataset import * 


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
TestGenerator = Completion_Network().to(device)
TestGenerator.load_state_dict(torch.load("./model_weight/Generator/epoch_0_weight.pth"))
dataset_test = PlaceTwoTestDataset()
rand_idx = np.random.randint(328500,size=1)[0]
unmasked_images, masked_images, masks, mask_center_x, mask_center_y = dataset_test[rand_idx]
unmasked_images= unmasked_images.view(1,3,256,256).to(device)
masked_images = masked_images.view(1,3,256,256).to(device)
masks = masks.view(1,1,256,256).to(device)

MaskAndMaskedImage = torch.cat([masked_images,masks],dim=1).to(device)
fake_image = TestGenerator(MaskAndMaskedImage).to(device)
print(fake_image)