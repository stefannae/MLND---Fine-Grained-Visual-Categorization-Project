import io
from PIL import Image
import torchvision.transforms as transforms


def get_transform(mode='train', norm=True, dataset='Dogs'):
    resize = transforms.Resize(256)
    rotation = transforms.RandomRotation(30)
    r_crop = transforms.RandomCrop(224)
    c_crop = transforms.CenterCrop(224)
    h_flip = transforms.RandomHorizontalFlip()
    tensor = transforms.ToTensor()
    
    if dataset == 'Dogs':
        normalize = transforms.Normalize(
            mean=(0.487, 0.459, 0.394),
            std=(0.229, 0.226, 0.224))
        
    else:
        # ImageNet values
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))
    
    augment = [rotation, r_crop, h_flip]
    
    transform_list = [resize]
   
    if mode == 'train' and norm == True:
        transform_list.extend(augment)
        transform_list.extend([tensor, normalize])
           
    elif mode in ['valid', 'test'] and norm == True:
        transform_list.extend([c_crop, tensor, normalize])
        
    elif norm == False:
        transform_list.extend([c_crop, tensor])
        
    else:
        print('Unrecognized mode!')
        
    return transforms.Compose(transform_list)


def transform_image(image_bytes):
    transform = get_transform('test')
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except:
        image = Image.fromarray(image_bytes)
    
    return transform(image).unsqueeze(0)
