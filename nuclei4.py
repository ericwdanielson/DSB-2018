"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random


# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/balloon"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from config import Config
import utils
import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class NucleiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5# Background + Stain_types

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 700
    VALIDATION_STEPS = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0
        
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    LEARNING_RATE = 0.001
    
    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    
    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400




############################################################
#  Dataset
############################################################

class NucleiDataset(utils.Dataset):

    
    
    def load_nuclei(self, dataset_dir, subset):
         # Add classes
        classes={}
        
        with open(os.path.join(dataset_dir, "via_region_data.json")) as j:
            annotations = json.load(j)
        image_names = list(annotations.keys())
        
        for name in image_names:
            image_info = annotations[name]
            classes.update({image_info['image_class']:0})
        
        #for i in classes.keys():
        self.add_class("nuclei", 1, "dapi")
        self.add_class("nuclei", 2, "nissl")
        self.add_class("nuclei", 3, "cresyl")
        self.add_class("nuclei", 4, "he")
        self.add_class("nuclei", 5, "dab")
            
        self.class_balancer = {"dapi":[],"nissl":[],"cresyl":[],"he":[],"dab":[]}
            
                
        for name in image_names:
            image_info = annotations[name]            
            if image_info['image_type'] == subset:
                image_path = dataset_dir + '/' + name[:-4] + '/images' + '/' + name
                image_id = name                
                class_id = image_info['image_class']                
                mask_path =  dataset_dir + '/' + name[:-4] + '/masks'
                self.add_image(
                        "nuclei",
                        image_id=image_id,  # use file name as a unique image id
                        path=image_path,
                        class_id = class_id,
                        mask_path=mask_path)
                self.filenames = self.filenames + [image_id]
                self.record_class(class_id,self.image_number)
                self.image_number = self.image_number + 1
               
                    
    def pre_load(self):
        '''for i in self.image_ids:
            self.my_images.append(self.load_image(i))
            m,i = self.load_mask(i)
            self.my_masks.append(m)
            self.my_mask_ids.append(i)
        self.preloaded=True'''
        
    def load_mask(self, image_id):
        """Loads masks from file
        """
        if self.preloaded:
            return np.copy(self.my_masks[image_id]),np.copy(self.my_mask_ids[image_id])
        # If not a nuclei dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        
        if image_info["source"] != "nuclei":
            return super(self.__class__, self).load_mask(image_id)

        
        # Get mask from png files in maskes dir set 255 to 1        
        mask_images = str(image_info['mask_path'] + '/*.png')
        mask = np.array(skimage.io.imread_collection(mask_images))
        #print(mask.shape)
        mask = np.transpose(mask,(1,2,0)) #need to change [n,height,width] to [height,width,n]
        mask = mask / 255
        
                
        # Map class names to class IDs.
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)        
        class_ids[:] = self.class_names.index(image_info["class_id"])
        #print("class : " + image_info["class_id"])

        # Return mask, and array of class IDs of each instance. 
        
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
   
    def record_class(self,class_id, image_index):
        print(class_id)
        self.class_balancer[class_id].append(image_index)
    
    def get_next_index(self):
        max = len(self.class_balancer.keys())
                
        if self.class_tracker >= max: #if at last class choose first non background
            self.class_tracker = 0
            
        image_list = list(self.class_balancer.values())[self.class_tracker]
        self.class_tracker = self.class_tracker + 1 #update to next class
        k = random.randint(0,len(image_list)-1)
        return image_list[k]
        


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleiDataset()
    dataset_train.load_nuclei(args.dataset, "train")
    dataset_train.prepare()
    #dataset_train.pre_load()
    print("loaded training")
    # Validation dataset
    
    dataset_val = NucleiDataset()
    dataset_val.load_nuclei(args.dataset, "train")
    dataset_val.prepare()
    #dataset_val.pre_load()
    print("loaded validation")
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='all')

def train_and_classify(model):
    keep_training = True
    while keep_training:
        #train(model)
        keep_training = evaluate_unclassified(model,0.4,"mrcnn_mask_loss")
        train(model)
        keep_training = False
    #train(model)

def evaluate_unclassified(model,threshold,monitor):
    '''This function evaluates all of the "unclassified images"
    the class of the unclassified should be the last class created and starts with class1
    the loss sores get stored in the annotations['filename']['class1'] loss info
    when the loss score is good (below 0.5 or 1) the image_type should be changed to "train" (one to "val")
    and the class_id updated to the latest class
    another round of trainging should be performed and new images evaluated. If no new images are added to the latest class
    a new class should be created from the first 'unclassified' found. It's class should be changed to the current class + 1 and it's type changed to 'train'
    all remaining 'unclassified' should have their class updated to the current class then the process should repeat.
    
    
    '''
    classes = {1:0}
    dataset = NucleiDataset()
    dataset.load_nuclei(args.dataset, "unclassified")
    dataset.prepare()
    #loading images of unknown class for evaluation and eventually class sorting
    print("Evaluating Unclassified images")
    scores = model.evaluate_unclassified(dataset,
                                         learning_rate =config.LEARNING_RATE,
                                         layers = 'heads',debug = False)
    
    annotations = json.load(open(os.path.join(args.dataset, "via_region_data.json")))
    filenames = list(scores.keys())
    no_images_moved = True
    new_validation_image = False
    new_training_image = False
    for f in filenames:
        if annotations[f]['image_type'] == 'unclassified':
            class_id = annotations[f]['image_class']
            if class_id in classes:
                classes[class_id] += 1
            else:
                classes.update({class_id:1})
            annotations[f]['eval_info'].update(scores[f])
            if float(annotations[f]['eval_info'][monitor]) < threshold:
                if new_training_image == True and new_validation_image == False:
                    annotations[f]['image_type'] = "train"
                    new_validation_image = True
                    new_training_image = True
                    no_images_moved = False
                else:
                    annotations[f]['image_type'] = "val"
                    no_images_moved = False
                    new_training_image = True
                                
    '''
    The following code is executed when no new images can be trained well with the existing classes
    in this case all remaining unclassified images get plased in a new class and the first unclassified images is used to train that class
    '''                            
    if no_images_moved: 
        train_image_found = False        
        for f in filenames:
            if annotations[f]['image_type'] == 'unclassified':
                if train_image_found == False:
                    annotations[f]['image_type'] = "train"
                    annotations[f]['image_class'] += 1
                    train_image_found = True
                else:
                    annotations[f]['image_class'] += 1
    #print("preparing to print")
    #exit()
    
    with open(os.path.join(args.dataset, "via_region_data.json"),'w') as f:
        json.dump(annotations,f, indent = 4)
    
    if no_images_moved and train_image_found == False:
        return False #This indicates no more evaluating needs to be done all images are placed in classes
    return True #there are more images that need to be evaluated
        
        
        
    
    

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    gray[:,:,:]=0
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash

def detect_and_save_mask(model,image_path,output_dir = 'output'):
    
    images = os.listdir(image_path)
        
    for image_name in images:
        image = skimage.io.imread(image_path + '/' + image_name)
        if image.shape[2] == 4:
            image = image[:,:,0:3]
        r = model.detect([image], verbose=1)[0]
        outpath = output_dir + '/' + image_name[:-4]
        if os.path.exists(outpath) == False:
            os.makedirs(outpath)
        skimage.io.imsave(outpath + '/' + image_name,image)
        masks = r['masks']
        for i in range(masks.shape[2]):
            mask = masks[:,:,i] * 255  
            if mask.shape[0] != 0:
                skimage.io.imsave(outpath + '/' + image_name[:-4] + '_m' + str(i) + '.png',mask)
            
    

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        if image.shape[2] == 4:
            image = image[:,:,0:3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect nuclei.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'splash' or eval")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/nuclei/dataset/",
                        help='Directory of the nuclei dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "eval":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleiConfig()   
    elif args.command == "eval":
        config = NucleiConfig() 
    else:
        class InferenceConfig(NucleiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    elif args.command == "eval":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_save_mask(model,image_path=args.image)
        #detect_and_color_splash(model, image_path=args.image,
                                #video_path=args.video)
    elif args.command == "eval":
        train_and_classify(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
