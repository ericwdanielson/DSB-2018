# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 12:48:08 2018

@author: ericw
"""

from imgaug import augmenters as iaa
import numpy as np
from skimage.io import imread, imsave, imread_collection
import os
import random
from PIL import Image

'''
Real time image augmentation slows the training down will augment the images and masks beforehand
Additionally will adjust so the classes are balanced.

read dir,

strip color channels and keep only blue (for now)
add some gaussian noise
flip lr
flip upd
crop and pad
gaussian noise
iaa.Scale((0.75,1.25))
iaa.CropAndPad(percent=(-0.25,0.25))
iaa.Fliplr(0.5)
iaa.Flipud(0.5)
iaa.GaussianBlur(sigma=(0.0, 3.0))
iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5)
iaa.MultiplyElementwise((0.5, 1.5))
iaa.Dropout(p=(0, 0.2))
iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})
iaa.PiecewiseAffine(scale=(0.01, 0.05))
546 bw
108 color

6,000 bw
6,000 color

2000 he, nissl, 6 tissue    

38 he
66 nissl
6 he tissue
'''
def create_image_and_mask_augmenter():
    aug = iaa.SomeOf((0, None), [
        iaa.Noop(),
        iaa.Scale((0.75,1.0)),
        iaa.CropAndPad(percent=(-0.25,0.25)), 
        iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)}),
        #iaa.PiecewiseAffine(scale=(0.01, 0.05)),
        iaa.Fliplr(1),
        iaa.Flipud(1)
        ])
    return aug

def create_image_only_augmenter():
    aug = iaa.SomeOf((0, None), [
        iaa.Noop(),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),
        iaa.AdditiveGaussianNoise(scale=0.02*255, per_channel=0.5),
        iaa.MultiplyElementwise((0.75, 1.25)),
        iaa.Dropout(p=(0, 0.1)),
        iaa.Add((-20, 20)),
        iaa.AddElementwise((-20, 20)),
        iaa.ContrastNormalization((0.75, 1.25))        
        ])
    return aug
def get_background(image,mask):
    sum = np.sum(mask,axis=2)
    if random.randint(0,1):
        back = image[sum==0,:]
    else:
        back = image[sum>0,:]
    return back #pixel values for background
    
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def get_forground(image,mask):
    boxes = extract_bboxes(mask)
    mini_images = []
    mini_masks = []
    i = 0
    for b in boxes:
        mini_images.append(image[b[0]:b[2],b[1]:b[3],:])
        mini_masks.append(mask[b[0]:b[2],b[1]:b[3],i])
        i += 1
    return mini_images, mini_masks, boxes

def scan_mask(image,mask,size):
    background = []
    mask_sum = np.sum(mask,axis=2)
    while len(background) == 0 and size > 2:
        size = int(size / 2)
        if(size < min(image.shape)):
            for h in range(0,image.shape[0],size):
                for w in range(0,image.shape[1],size):
                    if np.sum(mask_sum[h:h+size,w:w+size])==0:
                        background.append(image[h:h+size,w:w+size,:])
    return background, size

def make_new_image(image,mask,height,width,show,object_num,class_ids,method):
    size = 1
    if method == 1:
        background = get_background(image,mask)
    elif method == 2:
        background, size = scan_mask(image,mask,height)
        
    fi,fm,boxes = get_forground(image,mask)
    #mask_sum = np.sum(mask,axis=2)
    #object_num = 25
    new_image = np.zeros([height,width,3])
    new_mask = np.zeros([height,width,object_num])
    b_index = 0    
    for h in range(0,new_image.shape[0],size):
        for w in range(0,new_image.shape[1],size):
            if method == 1:
                new_image[h,w,:] = background[random.randint(0,len(background)-1)]
            elif method == 2:
                new_image[h:h+size,w:w+size,:] = background[random.randint(0,len(background)-1)]
            #new_image[h,w,:] = background[b_index]
            b_index += 1
            if b_index == len(background):
                b_index = 0
    
    for o in range(object_num):
        index = random.randint(0,len(fi)-1)
        stamp = fi[index]
        stamp_mask = fm[index]
        if random.randint(0,1):
            stamp = np.fliplr(stamp)
            stamp_mask = np.fliplr(stamp_mask)
        if random.randint(0,1):
            stamp = np.flipud(stamp)
            stamp_mask = np.flipud(stamp_mask)
        bw = boxes[index][3]-boxes[index][1]
        bh = boxes[index][2]-boxes[index][0]
        x = random.randint(0,width-bw)
        y = random.randint(0,height-bh)
        new_image[y:y+bh,x:x+bw,:][stamp_mask>0] = stamp[stamp_mask>0]
        new_mask[y:y+bh,x:x+bw,:][stamp_mask>0] = 0 #removes overlaps from previous
        new_mask[y:y+bh,x:x+bw,o] = stamp_mask #draws new
    
    if show:
        Image.fromarray(new_image.astype('uint8')).show()
    new_mask = np.moveaxis(new_mask,0,-1) 
    new_class_ids = np.ones([new_mask.shape[-1]], dtype=np.int32)
    new_class_ids[:] = class_ids[0]
    
    return new_image, new_mask, new_class_ids

def apply_image_and_mask_augmentation(image,mask,aug = None,swap_dim = True):
    if aug == None:
        aug = create_image_and_mask_augmenter()
    if swap_dim:
        mask = np.moveaxis(mask,0,-1)  
    merged = np.concatenate((image,mask),axis=2)
    merged_augmented = aug.augment_image(merged)
    images_aug = merged_augmented[:,:,0:3]
    masks_aug = merged_augmented[:,:,3:]
    if swap_dim:
        masks_aug = np.moveaxis(masks_aug,2,0)
    return images_aug,masks_aug

def apply_image_augmentation(image,aug = None):
    if aug == None:
        aug = create_image_only_augmenter()
    images_aug = aug.augment_image(image)
    return images_aug

def save_image(image,mask,output_dir,image_name,image_number):
    new_image_name = image_name[:-4] + '_'+str(image_number)
    if os.path.exists(output_dir + '/' + new_image_name) ==False:
        os.makedirs(output_dir + '/' + new_image_name + '/images')
        os.makedirs(output_dir + '/' + new_image_name + '/masks')
    img_dir = output_dir + '/' + new_image_name + '/images'
    mask_dir = output_dir + '/' + new_image_name + '/masks'
        
    imsave(img_dir + '/' + new_image_name +'.png',image)
    
    for i in range(mask.shape[0]):
        m = mask[i]
        if np.max(m) > 0:
            m[m>0] = 255
            imsave(mask_dir + '/' + str(i) +'.png',m)

def load_images(image_name,image_path,mask_path):
    image=imread(image_path +'/' + image_name)[:,:,0:3]
    masks = imread_collection(mask_path +'/' + image_name[:-4] + '/masks/*.png')
    return image,np.array(masks)

def process_image(image_name,image_path,mask_path,output_path,output_number,aug,aug_img):
    image,masks = load_images(image_name,image_path,mask_path)    
    for i in range(output_number):
        outi,outm = apply_image_and_mask_augmentation(np.copy(image),np.copy(masks),aug)
        outi = apply_image_augmentation(outi,aug_img)
        save_image(outi,outm,output_path,image_name,i+1)

def process_directory(image_path,mask_path,output_path,output_number,aug,aug_img):
    files = os.listdir(image_path)
    for f in files:
        process_image(f,image_path,mask_path,output_path,output_number,aug,aug_img)

def process_multi_directory(image_path,mask_path,output_path,output_number,aug,aug_img):
    dirs = os.listdir(image_path)
    for d in dirs:
        process_directory(image_path + '/' +d,mask_path,output_path,output_number,aug,aug_img)
        
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Augment Images')
        
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help='Image to apply the augmentation to')
    parser.add_argument('--dir', required=True,
                        metavar="path to dir of images",
                        help=' directory containing images to apply the augmentation to')
    parser.add_argument('--out', required=True,
                        metavar="path to dir of images",
                        help=' output dir')
    parser.add_argument('--mask', required=True,
                        metavar="path to parent dir containing masks",
                        help=' folder should contain subfolders with masks')
    parser.add_argument('--number', required=True,
                        metavar="number of images",
                        help='number of augmentations to create')
    parser.add_argument("--command",
                        metavar="<command>",
                        help="'image', 'dir' or multi_dir")
    
    args = parser.parse_args()
    aug = create_image_and_mask_augmenter()
    aug_img = create_image_only_augmenter()
    if args.command == "image":        
        process_image(args.image,args.dir,args.mask,args.out,int(args.number),aug,aug_img)
    if args.command == "dir":
        process_directory(args.dir,args.mask,args.out,int(args.number),aug,aug_img)
    if args.command == "multi_dir":
        process_multi_directory(args.dir,args.mask,args.out,int(args.number),aug,aug_img)