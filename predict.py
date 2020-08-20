import io
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models

from utils import transform_image
from model import Net2

use_cuda = torch.cuda.is_available()
if not use_cuda:
    map_location = "cpu"
else:
    map_location = "cuda"

# Dog breed clasifier
model = Net2()
model.load_state_dict(torch.load('model_transfer.pt', map_location=map_location))
model.eval()

if use_cuda:
    model.cuda()

with open("dog_classes.txt", "r") as f:
    class_names = f.readlines()

class_names = [name[:-1] for name in class_names]

# Face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# Generic classifier that can detect dogs and humans 
VGG16 = models.vgg16(pretrained=True)
VGG16.eval()

if use_cuda:
    VGG16.cuda()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    tensor = tensor.to(device)
    
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    
    return class_names[predicted_idx]


def face_detector(image_bytes, box=False):
    if isinstance(image_bytes, str):
        with open(image_bytes,'rb') as f:
            image = f.read()

        image_bytes = image
    
    image = np.asarray(bytearray(image_bytes), dtype="uint8")
    gray = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    faces = face_cascade.detectMultiScale(gray)
    if not box:
        return len(faces) > 0
    else:
        return faces


def VGG16_predict(image_entry):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
 
    transform = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
    
    if isinstance(image_entry, str):
        with open(image_entry,'rb') as f:
            image = Image.open(image_entry).convert('RGB')
           
    else:
        image = Image.open(io.BytesIO(image_entry))
        
    image = transform(image)
    
    device = torch.device("cuda" if use_cuda else "cpu")
    image = image.to(device)
    
    return np.argmax(VGG16(image.unsqueeze(0)).detach().cpu().numpy()) # predicted class index


def dog_detector(image): 
    img_class = VGG16_predict(image)
    
    if 151 <= img_class <= 288:
        return True
    else:
        return False
