import numpy as np
from matplotlib import pyplot as plt
import re
import cv2
import torch
import models
import random
import json

import albumentations as A
import matplotlib.pyplot as plt
import kornia as K
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from datasets import register
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

@register('Ocr_images_lp')
class Ocr_images_lp(Dataset):
    def __init__(
            self,
            alphabet,
            k,
            imgW,
            imgH,
            aug,
            image_aspect_ratio,
            background,
            with_lr = False,
            test = False,
            dataset=None,
            ):
        
        self.imgW = imgW
        self.imgH = imgH
        self.aug = False
        self.ar = image_aspect_ratio
        self.background = eval(background)
        self.test = test
        self.dataset = dataset
        self.k = k
        self.alphabet = alphabet
        self.with_lr = with_lr
        self.transformImg = np.array([
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True, p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, always_apply=True, p=1.0),
                A.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)}, rotate=(-10, 10), shear={'x': (-10, 10), 'y': (-10, 10)}, mode=cv2.BORDER_CONSTANT, cval=self.background, fit_output=True, keep_ratio=True, p=1.0, always_apply=True),
                A.SafeRotate(limit=15, value=(127, 127, 127), border_mode=cv2.BORDER_CONSTANT, p=1.0, always_apply=True),

                A.Posterize(num_bits=4, always_apply=True, p=1.0),
                A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=True, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1.0),
                
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True, p=1.0),
                A.PixelDropout(dropout_prob=0.01, per_channel=True, drop_value=0, mask_drop_value=None, always_apply=True, p=1.0),
                A.ImageCompression(quality_lower=90, quality_upper=100, always_apply=True, p=1.0),
                A.ColorJitter(p=1),
                None
            ])
            
    def Open_image(self, img, cvt=True):
        # print(img)
        img = cv2.imread(img)
        if cvt is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def padding(self, img, min_ratio, max_ratio, color = (0, 0, 0)):
        img_h, img_w = np.shape(img)[:2]

        border_w = 0
        border_h = 0
        ar = float(img_w)/img_h

        if ar >= min_ratio and ar <= max_ratio:
            return img, border_w, border_h

        if ar < min_ratio:
            while ar < min_ratio:
                border_w += 1
                ar = float(img_w+border_w)/(img_h+border_h)
        else:
            while ar > max_ratio:
                border_h += 1
                ar = float(img_w)/(img_h+border_h)

        border_w = border_w//2
        border_h = border_h//2

        img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
        
        return img, border_w, border_h
    
    def extract_plate_numbers(self, file_path, pattern):
        # List to store extracted plate numbers
        plate_numbers = []
        
        # Open the text file
        with open(file_path, 'r') as file:
            # Iterate over each line in the file
            for line in file:
                # Search for the pattern in the current line
                matches = re.search(pattern, line)
                # If a match is found
                if matches:
                    # Extract the matched string
                    plate_number = matches.group(1)
                    # Add the extracted plate number to the list
                    plate_numbers.append(plate_number)
        
        # Return the list of extracted plate numbers
        return plate_numbers[0]
    
    def rectify_img(self, img, pts, margin=2):
        # obtain a consistent order of the points and unpack them individually
        # rect = order_points(pts)
        (tl, tr, br, bl) = pts
     
        # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
     
        # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        maxWidth += margin*2
        maxHeight += margin*2
     
        # now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
        ww = maxWidth - 1 - margin
        hh = maxHeight - 1 - margin
        c1 = [margin, margin]
        c2 = [ww, margin]
        c3 = [ww, hh]
        c4 = [margin, hh]

        dst = np.array([c1, c2, c3, c4], dtype = 'float32')
        pts = np.array(pts, dtype='float32')
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
     
        return warped 
    
    def padding(self, img, min_ratio, max_ratio, color = (0, 0, 0)):
        img_h, img_w = np.shape(img)[:2]

        border_w = 0
        border_h = 0
        ar = float(img_w)/img_h

        if ar >= min_ratio and ar <= max_ratio:
            return img, border_w, border_h

        if ar < min_ratio:
            while ar < min_ratio:
                border_w += 1
                ar = float(img_w+border_w)/(img_h+border_h)
        else:
            while ar > max_ratio:
                border_h += 1
                ar = float(img_w)/(img_h+border_h)

        border_w = border_w//2
        border_h = border_h//2

        img = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value = color)
        
        return img, border_w, border_h
    
    def get_pts(self, file):
        file = file.with_suffix('.json')
        with open(file, 'r') as j:
            pts = json.load(j)['shapes'][0]['points']
        
        return pts
    
    def collate_fn(self, datas):
        imgs = []
        gts = []
        name = []
        for item in datas:
            # name = item['img']
            name.append(item['img'])
            
            
            
            if self.with_lr:
                img = self.Open_image(item["img"].replace('HR', 'LR') if random.random() < 0.5 else item["img"])
            else:
                img = self.Open_image(item['img'])
            if self.aug is True:
                augment = np.random.choice(self.transformImg, replace=True)
                rectify_assert = random.random()                
                if rectify_assert < 0.5 and "dataset_intelbras" in str(item['img']):
                    img = self.rectify_img(img, self.get_pts(Path(item['img'])), margin=2)    
            
                if augment is not None:
                    img = augment(image=img)["image"]
            
                 
            
            
            img, _, _ = self.padding(img, self.ar-0.15, self.ar+0.15, self.background)    
            img = resize_fn(img, (self.imgH, self.imgW))
            imgs.append(img)
            gt = self.extract_plate_numbers(Path(item["img"]).with_suffix('.txt'), pattern=r'plate: (\w+)')
            gts.append(gt)  
        
        batch_txts = gts
        
        batch_imgs = torch.stack(imgs)
        
        return {
                'img': batch_imgs, 'text': batch_txts, 'name': name
        }
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
