import numpy as np
from PIL import Image
import os
from get_color_util import get_colormap
from keras.preprocessing.image import *
from skimage.color import *
from collections import defaultdict
import matplotlib.pyplot as plt



def comp_posterior(max_files = 10,data_dir = "./VOCdevkit/VOC2012/JPEGImages/"):
    NUM_CLASSES = 313
    #data_dir = "./VOCdevkit/VOC2012/JPEGImages/"
    data_dir = '/home/zc2388/dataset/flowers/jpg/'
    iter = 0
    #ab_label_map = get_colormap()
    #bins = ab_label_map.keys()
    cnt = defaultdict(int)
    #cat_idx_file = '/home/zc2388/class_wise_label/cat_train.txt'
    #f = open(cat_idx_file)
    ##fdata = f.readlines()
    #.close()
    
    #for i in fdata:
	#cat_file_path = data_dir + i[:11] + '.jpg'
	#print(cat_file_path)
    for file_ in os.listdir(data_dir):

        image = Image.open(data_dir + file_)
	#image = Image.open(cat_file_path)
        image = img_to_array(image)
        image = image.astype("uint8")
	print(np.min(image), np.max(image))
	ab_img = rgb2lab(image)[:,:,1:]
	ab_img = ab_img + 5.0*np.ones(ab_img.shape)
        image = (ab_img/10).astype(int) #maybe tune this?
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
		if not (image[i,j,0],image[i,j,1]) in cnt:
			cnt[(image[i,j,0],image[i,j,1])] =0
                cnt[(image[i,j,0],image[i,j,1])] += 1
        if iter==max_files:
            break
        iter += 1
	if iter%1000 ==0:
            print(iter)
    print(cnt)
    total = float(sum(cnt.values()))
    print(total)
    ab_label_map = get_colormap()
    prob = defaultdict(float)
    res = np.zeros(NUM_CLASSES)
    for x,y in ab_label_map.keys():
        prob[(x//10,y//10)] = float(cnt[(x//10,y//10)])/total
        res[ab_label_map[(x,y)]] = cnt[(x//10,y//10)]/total

    np.save("./models/posterior_prob_flower.npy",res)

    test_colormap = np.ones((23, 23, 3))
    for i in range(23):
        for j in range(23):
            test_colormap[i, j, 0] = prob[((i-11),(j-11))]*2000
    lab_rgb_image = lab2rgb(test_colormap)
    plt.imshow(lab_rgb_image)
    #plt.show()



if __name__=="__main__":
    """ 
    filename = "./VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"
    image = Image.open(filename)
    image = img_to_array(image)
    image = image.astype("uint8")
    """
    #data_dir = "./VOCdevkit/VOC2012/JPEGImages/"
    data_dir = my_dir = os.path.expanduser("~/dataset/flowers/jpg") +'/'# This data_dir differs from original one! / at end.
    #data_dir = "./"
    comp_posterior(max_files = 20000,data_dir = data_dir)




