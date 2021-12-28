from tensorboardX import SummaryWriter
import numpy as np
import cv2
def tensor2image(tensor_image):
  output_image = ((tensor_image.cpu().permute(1,2,0).detach().numpy()*0.5+0.5)*255).astype(np.int32)
  clipped_image = np.clip(output_image,0,255)
  # print(clipped_image.shape)
  # print(clipped_image)
  return clipped_image

def DrawMiddleTrainResult(writer,origin_image, masked_image, return_image,epoch):
  #1epoch당 변화하는걸 찍어도 좋을듯하다. 
  
  origin_image = tensor2image(origin_image)
  # writer.add_image('real image',origin_image)
  
  cv2.imwrite("train_image/real image"+str(epoch)+".jpg",origin_image)

  masked_image = tensor2image(masked_image)
  # writer.add_image('masked image',masked_image)
  cv2.imwrite("train_image/mask image"+str(epoch)+".jpg",masked_image)
  return_image = tensor2image(return_image)
  # writer.add_image('fake image',return_image)
  cv2.imwrite("train_image/fake image"+str(epoch)+".jpg",return_image)


def DrawMiddleTestResult(writer,origin_image, masked_image, return_image,epoch):
  origin_image = tensor2image(origin_image)
  # writer.add_image('real image',origin_image)
  
  cv2.imwrite("test_image/real image"+str(epoch)+".jpg",origin_image)

  masked_image = tensor2image(masked_image)
  # writer.add_image('masked image',masked_image)
  cv2.imwrite("test_image/mask image"+str(epoch)+".jpg",masked_image)

  return_image = tensor2image(return_image)
  # writer.add_image('fake image',return_image)
  cv2.imwrite("test_image/fake image"+str(epoch)+".jpg",return_image)