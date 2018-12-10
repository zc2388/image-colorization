import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import time
#import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import keras.backend as K

from models import *
from skimage.color import *
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util
import scipy as sp

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
from utils.ColorizationDataGenerator import *
import time

import matplotlib.pyplot as plt
from keras.losses import mean_squared_error

from get_color_util import *

source_dir = "/home/zc2388/segmentation/image-colorization"
model_name = 'Colorization_As_Segmentation_AtrousFCN_Resnet50_16s'
out_dir = '/home/zc2388/cat_prediction_out/'

#checkpoint_path = "/home/zc2388/segmentation/image-colorization/Models/cat_lambda_0.25/model.hdf5"
#Colorization_As_Segmentation_AtrousFCN_Resnet50_16s/checkpoint_weights.hdf5

checkpoint_path = source_dir + "/Models/" +model_name + "/model.hdf5"

batch_shape = (1, 320, 320, 3)
model = globals()[model_name](batch_shape=batch_shape, input_shape=(320, 320, 3))
model.load_weights(checkpoint_path, by_name=True)

train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
# train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
data_suffix='.jpg'
label_suffix='.png'

batch_size =1

target_size = (320, 320)


fp = open('/home/zc2388/class_wise_label/cat_val.txt')
lines = fp.readlines()
fp.close()


if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

target_path = "/home/zc2388/741754417.png" 
only_target = False

for i in range(100):
    jpgid = ''
    if only_target:
        x = Image.open(target_path)
        jpgid = 'testing'
    else:
            jpgid = lines[i].strip('\n')
    
            x = Image.open("/home/zc2388/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/" + jpgid + ".jpg")
        if not os.path.isdir(out_dir+jpgid ):
            os.mkdir(out_dir+jpgid )
    x = x.resize((320,320))
    x.save(out_dir+jpgid +'/' + jpgid + 'read.png','png')
    x = img_to_array(x)

        if x.shape[0]<320 or x.shape[1]<320:
            continue

        x = x[0:320,0:320,:]
        lab_image = rgb2lab(x/255)
        lab_rgb_image = rgb2lab(x/255)
        lab_rgb_image[:,:,1] = np.zeros((x.shape[0],x.shape[1]))
        lab_rgb_image[:,:,2] = np.zeros((x.shape[0],x.shape[1]))
        image = lab2rgb(lab_rgb_image)

        image = image*255
        image1 = np.expand_dims(image, axis=0)
        image2 = preprocess_input(image1)
        result1 = model.predict(image2, batch_size=1)
    result = np.argmax(np.squeeze(result1), axis=-1).astype(np.uint8)

    count_dict = {}
    for i in range(320):
            for j in range(320):
                if not result[i][j] in count_dict:
                        count_dict[result[i][j]] = 0
                count_dict[result[i][j]]+=1
    print(count_dict)
    print("=============================================")

    #np.set_printoptions(precision=4)
    #np.save(out_dir + jpgid +'_label', result)
    #np.save(out_dir + jpgid +'_raw_prediction', result1)

    #lab_image[:,:,1] = result1[0,:,:,0]
    #lab_image[:,:,2] = result1[0,:,:,1]
    #recolor_ab_channel = pred_2_image_5nn(result1[0,:,:,:], lab_image[:,:,0])    
    #label_result =  np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
    #for i in range(320):
    #    for j in range(320):
    #
    #        lab_image[i,j,1] = result[i,j]
    #        lab_image[i,j,2] = result[i,j]
    recolor_ab_channel = pred_2_image_5nn(result1[0,:,:,:], lab_image[:,:,0])
    result_rgb = lab2rgb(recolor_ab_channel)
    j = Image.fromarray((result_rgb * 255).astype(np.uint8))
    j.save(out_dir + jpgid+ '/'+ jpgid +'_5nn.png', "png")
    
    recolor_ab_channel = pred_2_image_mode(result1[0,:,:,:], lab_image[:,:,0])
    result_rgb = lab2rgb(recolor_ab_channel)
    j = Image.fromarray((result_rgb * 255).astype(np.uint8))
    j.save(out_dir + jpgid+ '/'+ jpgid +'_mode.png', "png")

    recolor_ab_channel = pred_2_image_anneal(result1[0,:,:,:], lab_image[:,:,0])
    result_rgb = lab2rgb(recolor_ab_channel)
    j = Image.fromarray((result_rgb * 255).astype(np.uint8))
    j.save(out_dir + jpgid+ '/'+ jpgid +'_anneal.png', "png")

    # plt.imshow(result_rgb)
    # plt.show()


