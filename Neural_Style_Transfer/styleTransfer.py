
# Loading required libraries

import torch
from torchvision import models
from PIL import Image
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


def load_vgg_model():  
  vgg_model = models.vgg19(pretrained=True)  
  return vgg_model


def get_vgg_features(vgg_model):  
  return vgg_model.features


# def freeze_vgg_parameters(vgg_model):
#   for parameters in vgg_model.parameters():
#     parameters.requires_grad_(False)




"""**In NST we do not need classifier part of vgg model, we only need feature extration part bcz from feature extration part we will get feature of content and style images**

**So we will remove the classifier part from vgg_model**
"""



# function to load image using image path, preprocess it and return preprocessed image
def preprocess(img_path, max_size = 224):
  image = Image.open(img_path)
  image = image.convert('RGB')

  if max(image.size) > max_size:
    size = max_size

  else:
    size = max(image.size)
  
  # now applying transformer
  img_transforms = T.Compose([
                              T.Resize(size),
                              T.ToTensor(),   # (224,224,3) ---> (3,224,224)
                              T.Normalize(mean=[0.485, 0.463, 0.406],
                                          std=[0.229, 0.224,0.225])
                    
  ])

  image = img_transforms(image)

  # now unsquuezing dimension at axis 0, bcz we are going to
  # ... add batch size, bcz model input are in shape
  # .... (batch_size, channel, height, width)

  image = image.unsqueeze(0)  # (3, 224,224) ---> (1,3,224,224)

  return image



# function to deprocess image using tensor, and return deprocessed image
def deprocess(tensor):
  image = tensor.to('cpu').clone()
  image = image.numpy()
  image = image.squeeze(0) # (1,3,224,224) --> (3, 224, 224)
  image = image.transpose(1,2,0)  # (3,224,224) --> (224,224,3)
  image = image * np.array([0.229,0.224,0.225]) + np.array([0.485, 0.463, 0.406])  # image * std + mean
  image = image.clip(0,1)

  return image



"""### **Getting content, style features and create gram matrix**"""
def get_features(image, model):

  layers = {
      '0' : 'conv1_1',
      '5' : 'conv2_1',
      '10' : 'conv3_1',
      '19' : 'conv4_1',
      '21' : 'conv4_2',   # content feature
      '28' : 'conv5_1'
      
  }

  x = image

  Features = {}

  for name, layer in model._modules.items():
    x = layer(x)

    if name in layers:
      Features[layers[name]] = x
  
  return Features



# function for defining content image loss
def content_loss(target_conv4_2, content_conv4_2):
  loss = torch.mean((target_conv4_2 - content_conv4_2))
  
  return loss

# now computing gram matrix
def gram_matrix(tensor):
  b,c,h,w = tensor.size()
  tensor = tensor.view(c,h*w)
  gram_matrix = torch.mm(tensor, tensor.t())   # matrix maltiplication of tensor with transpose of tensor

  return gram_matrix


# function for defining style image loss
def style_loss(style_weights, target_features, style_grams):

  loss = 0

  for layer in style_weights:
    target_f = target_features[layer]
    target_gram = gram_matrix(target_f)
    style_gram = style_grams[layer]
    b,c,h,w = target_f.shape
    layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

    loss += layer_loss/(c*h*w)

    return loss

# now defining total loss
def total_loss(c_loss, s_loss, alpha, beta):
  loss = alpha*c_loss + beta*s_loss
  return loss


# setting hyperparameters

alpha = 1     # content image weight
beta = 1e5    # style image weight
epochs = 100
show_after_every = 20


def generateNeuralStyleImage(content_img_path, style_img_path):  

  # loading vgg model
  vgg_model = load_vgg_model()

  # get only feature part from model
  vgg_model = get_vgg_features(vgg_model)

  # freezing model layers
  # vgg_model = freeze_vgg_parameters(vgg_model)

  for parameters in vgg_model.parameters():
    parameters.requires_grad_(False)

  # setting device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # preprocess image
  content_p = preprocess(content_img_path)
  style_p = preprocess(style_img_path)

  # moving preprocessed image to device
  content_p = content_p.to(device)
  style_p = style_p.to(device)

  # Getting content, style features and create gram matrix

    #  now loading content and style features
  content_f = get_features(content_p, vgg_model)
  style_f = get_features(style_p, vgg_model)

    # setting style grams
  style_grams = { layer : gram_matrix( style_f[layer] ) for layer in style_f }


  # Creating Style and Content Loss function

    # for style loss we need to define style weight
  style_weights = {
      'conv1_1' : 1.0,
      'conv2_1' : 0.75,
      'conv3_1' : 0.2,
      'conv4_1' : 0.2,
      'conv5_1' : 0.2
  }

  # ------------------------
  # In Process / TODO
  # ------------------------


  # defining target/generated image variable
  target = content_p.clone().requires_grad_(True).to(device)
  target_f = get_features(target, vgg_model)


  # setting optimizer
  optimizer = optim.Adam([target], lr = 0.003)
    

  # defining training loop for optimizing target image for better style transfer
  results = []

  for epoch in range(epochs):
    target_f = get_features(target, vgg_model)
    c_loss = content_loss(target_f['conv4_2'], content_f['conv4_2'])
    s_loss = style_loss(style_weights, target_f, style_grams)
    t_loss = total_loss(c_loss, s_loss, alpha, beta)

    optimizer.zero_grad()   # zero_grad clears old gradients from the last step
    t_loss.backward()    # used to compute the gradient during the backward pass in a neural network
    optimizer.step()    # performs a parameter update based on the current gradient

    if epoch % show_after_every == 0:
      print(f"Total Loss at Epoch {epoch} : {t_loss}")
      detached_target = target.detach()
      deprocessed_img = deprocess(detached_target)
      results.append(deprocessed_img)

  # returning target image
  detached_target = target.detach()
  generated_img = deprocess(detached_target)

  # print(f"Shape of generated image is: {generated_img.shape}\n")
  # print(f"Type of generated image is: {type(generated_img)}\n")

  # plt.figure(figsize=(10,7))
  # plt.imshow(generated_img)
  # plt.show()
  # plt.axis('off');


  return generated_img

  
# if __name__ == '__main__':
#   generateNeuralStyleImage('./face.jpg', './flower3.jpg')

  





