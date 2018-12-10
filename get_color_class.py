import numpy as np
import matplotlib.pyplot as plt
from skimage.color import *
import os
from PIL import Image
from keras.preprocessing.image import *

test_colormap = np.zeros((23,23,3))
test_colormap[:,:,0] = np.ones((23,23))*50

for i in range(23):
    for j in range(23):
        test_colormap[i,j,1] = (i-11)*10.0
        test_colormap[i,j,2] = (j-11)*10.0

#lab_rgb_image = lab2rgb(test_colormap)
#plt.imshow(lab_rgb_image)
#plt.show()

valid_color_idx = {}
for i in range(21):
    valid_color_idx[i] = []

valid_color_idx[0] =[]
valid_color_idx[1] =[]
valid_color_idx[2] =[20,19,18,17,16]
valid_color_idx[3] =[20,19,18,17,16,15,14,13]
valid_color_idx[4] =[20,19,18,17,16,15,14,13,12,11]
valid_color_idx[5] =[20,19,18,17,16,15,14,13,12,11,10,9]
valid_color_idx[6] =[21,20,19,18,17,16,15,14,13,12,11,10,9,8]
valid_color_idx[7] =[21,20,19,18,17,16,15,14,13,12,11,10,9,8,7]
valid_color_idx[8] =[21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6]
valid_color_idx[9] =[21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6]

valid_color_idx[10] =[21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5]
valid_color_idx[11] =[21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4]

valid_color_idx[12] =[20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3]
valid_color_idx[13] =[20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3]
valid_color_idx[14] =[20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
valid_color_idx[15] =[20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
valid_color_idx[16] =[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
valid_color_idx[17] =[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
valid_color_idx[18] =[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
valid_color_idx[19] =[18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
valid_color_idx[20] =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
valid_color_idx[21] =[2,3,4,5,6,7,8,9,10,11]
valid_color_idx[22] =[]
for i in range(23):
    valid_color_idx[i] = set(valid_color_idx[i])

ab_label_map = {}
ab_label_hashmap = {}
label_idx = 0
for i in range(23):
    for j in range(23):
        if j not in valid_color_idx[i]:
            test_colormap[i,j,0] = 100
            test_colormap[i,j,1] = 0
            test_colormap[i,j,2] = 0
        else:
            ab_label_map[(int((i-11)*10), int((j-11)*10))] = label_idx
            ab_label_hashmap[(float(i)-11.0)*10.0 + (float(j)-11.0)*10.0/256.0] = label_idx
            label_idx+=1

for i in range(23):
    for j in range(23):
        if j not in valid_color_idx[i]:
            min_idx = -1
            min_dis = 9999
            for si in range(23):
                for sj in range(23):
                    if (si-i)**2 + (sj-j)**2<min_dis and (sj in valid_color_idx[si]):
                        min_idx = ab_label_map[(int((si-11)*10), int((sj-11)*10))]
                        min_dis = (si-i)**2 + (sj-j)**2
            
            
            ab_label_map[(int((i-11)*10), int((j-11)*10))] = min_idx
            ab_label_hashmap[(float(i)-11.0)*10.0 + (float(j)-11.0)*10.0/256.0] = min_idx
            

#lab_rgb_image = lab2rgb(test_colormap)
#plt.imshow(lab_rgb_image)
#plt.show()

print(ab_label_hashmap)
for i in range(23):
        print(ab_label_hashmap[(i-11.0)*10.0/256.0])


def get_label_array(image_path, ab_label_hashmap):
    x = Image.open(image_path)
    x = img_to_array(x)
    lab_image = rgb2lab(x/255)
    ab_image = lab_image[:,:,1:]
    ab_image_round = (10.0*((ab_image+5)/10.0).astype(int)).astype(float)
    ab_image_round[:,:,1] /= 256.0
    ab_image_hash = np.sum(ab_image_round, axis = 2)

    label_map = -1* np.ones(ab_image_hash.shape)

    for i in ab_label_hashmap:
        current_val = i
        current_label = ab_label_hashmap[i]
        this_label_idx_mask = (ab_image_hash == i)
        label_map[this_label_idx_mask] = current_label

    if np.min(label_map)==-1:
        print("Noob!")
        print(np.max(label_map))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if label_map[i,j] == -1:
                    print(ab_image[i,j])
    return label_map

def process_data(data_dir, output_folder, data_idx_file_path, data_suffix):
    fp = open(data_idx_file_path)
    lines = fp.readlines()
    fp.close()

    count  = 0
    for line in lines:
        line = line.strip('\n')
        file_path = os.path.join(data_dir, line+data_suffix)

        destination_path = os.path.join(output_folder, line+'.npy')
        label_map_matrix = get_label_array(file_path, ab_label_hashmap)

        np.save(destination_path, label_map_matrix)
        count+=1
        if count%1000 == 0:
            print( str(count) + " images processed")


        #print(file_path + " converted!")

train_file_path = os.path.expanduser('~/dataset/flowers/jpg/files.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
# train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
#data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
data_dir        = os.path.expanduser('~/dataset/flowers/jpg')
output_folder = os.path.expanduser('~/dataset/flowers/color_label')
data_suffix=''
label_suffix='.npy'
classes = 313

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

print(len(ab_label_hashmap))

process_data(data_dir, output_folder, train_file_path, data_suffix)
#process_data(data_dir, output_folder, val_file_path, data_suffix)

