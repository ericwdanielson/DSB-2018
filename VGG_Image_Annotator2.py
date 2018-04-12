# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from skimage import measure
#from skimage.io import imread
import json
import os
import argparse

argparser = argparse.ArgumentParser(
    description='VGG Image Style annotate images and place annotations in train directory')

argparser.add_argument(
    '-c',
    '--command',
    help='create or update_classes ')

argparser.add_argument(
    '-i',
    '--input',
    help='root directory of images and masks')

argparser.add_argument(
    '-o',
    '--annot_dir',
    help='annotation directory')

argparser.add_argument(
    '-m',
    '--merge',
    help='annotation directory')


# Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 28503151_5b5b7ec140_b.jpg':{
        #   'image_type':"train/val/unclassified",
        #   'mask_path': "path_to_mask_root",
        #   'image_class':1,
        #   'eval_info':{"rpn_class_loss":#, "rpn_bbox_loss":#, "mrcnn_class_loss":#, "mrcnn_bbox_loss":#, "mrcnn_mask_loss:#"}
        #
        # }
        # We mostly care about the x and y coordinates of each region
        # to find these images it should be root_dir->image_name_root->images/masks->
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_image_info():
    image_type = {'image_type':'train'}    
    image_class = {'image_class':1} # by default everything is class1
    eval_info = {"eval_info":{"loss":100,"rpn_class_loss":100, "rpn_bbox_loss":100, "mrcnn_class_loss":100, "mrcnn_bbox_loss":100, "mrcnn_mask_loss":100}}    
    image_info = {}
    image_info.update(image_type)    
    image_info.update(image_class)
    image_info.update(eval_info)
    return image_info
    
def VGG_Image_Annotator(image_name):
    filename = image_name
    image_info = get_image_info()    
    annotation = {filename:image_info}
    return annotation

def write_annotation(annotation,annotation_path,filename):
    
    filename = annotation_path + '/' + filename[:-5] + '.json'
    with open(filename,'w') as f:
        json.dump(annotation,f, indent = 4)

def convert_masks_to_annotation(path_to_images,annotation_path,write):    
    path_to_image = path_to_images + '/images/'
    image_name = os.listdir(path_to_image)[0]
    annotation = VGG_Image_Annotator(image_name)
    if write:
        write_annotation(annotation,annotation_path,image_name)
    else:
        return annotation
    
def batch_annotate_images(path_to_parent_directory,path_to_annotation_directory):
    #directories = listdir_fullpath(path_to_parent_directory)
    directories = os.listdir(path_to_parent_directory)
    for dir in directories:
        if os.isdir(dir):
            path = path_to_parent_directory + '/' + dir        
            convert_masks_to_annotation(path,path_to_annotation_directory,write = True)
        
def batch_annotate_images_merge(path_to_parent_directory,path_to_annotation_directory):
    #directories = listdir_fullpath(path_to_parent_directory)
    directories = os.listdir(path_to_parent_directory)
    annotations = {}
    num = 0
    for dir in directories:
        path = path_to_parent_directory + '/' + dir        
        annot = convert_masks_to_annotation(path,path_to_annotation_directory,write = False)
        annotations.update(annot) #keys should be filenames
        #print(annotations['filename'])
        num = num + 1        
    write_annotation(annotations,path_to_annotation_directory,'via_region_data.json')

def update_class_info(class_dir,annotation_dir):
    data = json.load(open(annotation_dir + '/' + 'via_region_data.json'))
    classes = os.listdir(class_dir)
    for c in classes:
        files = os.listdir(class_dir + '/' + c)
        for f in files:
            for key in data.keys():
                if key.startswith(f[:-4]):                
                    data[key].update({'image_class':c})
    write_annotation(data,annotation_dir,'via_region_data.json')
        
def _main_(args):
    path_to_parent_directory = args.input
    path_to_annotation_directory = args.annot_dir
    if args.command == "update_classes":
        update_class_info(path_to_parent_directory,path_to_annotation_directory)
    elif args.merge == 'False':
        batch_annotate_images(path_to_parent_directory,path_to_annotation_directory)
    else:
        batch_annotate_images_merge(path_to_parent_directory,path_to_annotation_directory)
    print("done")
    
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
