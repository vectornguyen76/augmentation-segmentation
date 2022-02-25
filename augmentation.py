import random
import os 
from glob import glob
import cv2
from matplotlib import pyplot as plt
import albumentations as A


def load_image(path_image, path_mask, size=(256, 256)):
    image = cv2.imread(path_image)
    image = cv2.resize(image, size)
    
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size)
    
    return image, mask
    

def save_images(image, mask, path_image, path_mask, i):
    path_image = os.path.join(path_image, f"{i}.jpg")
    path_mask = os.path.join(path_mask, f"{i}.jpg")
    cv2.imwrite(path_image, image)
    cv2.imwrite(path_mask, mask)


def HorizontalFlip(image, mask, path_image, path_mask, i):
    aug = A.HorizontalFlip(p=1)

    augmented = aug(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i
    i+=1
    return i

def VerticalFlip(image, mask, path_image, path_mask, i):
    aug = A.VerticalFlip(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def RandomRotate90(image, mask, path_image, path_mask, i):
    aug = A.RandomRotate90(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def Rotate(image, mask, path_image, path_mask, i):
    aug = A.Rotate(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def Transpose(image, mask, path_image, path_mask, i):
    aug = A.Transpose(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i


def Compose(image, mask, path_image, path_mask, i):
    aug = A.Compose([
        A.VerticalFlip(p=1),              
        A.Rotate(p=1)])

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i


def Blur(image, mask, path_image, path_mask, i):
    aug = A.Blur(blur_limit=3, p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def HueSaturationValue(image, mask, path_image, path_mask, i):
    aug = A.HueSaturationValue(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def GridDistortion(image, mask, path_image, path_mask, i):
    aug = A.GridDistortion(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def CLAHE(image, mask, path_image, path_mask, i):
    aug = A.CLAHE(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i

def RandomBrightnessContrast(image, mask, path_image, path_mask, i):
    aug = A.RandomBrightnessContrast(p=1)

    augmented = aug(image=image, mask=mask)

    image = augmented['image']
    mask = augmented['mask']
    
    save_images(image, mask, path_image, path_mask, i)
    i+=1
    return i



def main():
    # Path input read image
    path_input_images = glob("NA_datasets/*.jpg")
    
    # Path input read label
    path_input_masks = glob("GT_datasets/*.jpg")

    # path save image and label
    path_image = "datasets/image"
    path_mask = "datasets/label"

    # Create path save image and label
    os.makedirs("datasets", exist_ok=True)
    os.makedirs(path_image, exist_ok=True)
    os.makedirs(path_mask, exist_ok=True)

    # Create set point name image
    i = 0

    for j in range(len(path_input_images)):
        # Load image
        image, mask = load_image(path_input_images[j], path_input_masks[j], (512,512))
        
        # Augmentation 
        i = VerticalFlip(image, mask, path_image, path_mask, i)
        i = RandomRotate90(image, mask, path_image, path_mask, i)
        i = Rotate(image, mask, path_image, path_mask, i)
        i = Transpose(image, mask, path_image, path_mask, i)
        i = Compose(image, mask, path_image, path_mask, i)
        i = Blur(image, mask, path_image, path_mask, i)
        i = HueSaturationValue(image, mask, path_image, path_mask, i)
        i = GridDistortion(image, mask, path_image, path_mask, i)
        i = CLAHE(image, mask, path_image, path_mask, i)
        i = RandomBrightnessContrast(image, mask, path_image, path_mask, i)
    
    print("Successful !")
    
        
if __name__== "__main__" :
    main()