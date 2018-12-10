import numpy as np
import matplotlib.pyplot as plt
from skimage.color import *
import os
from PIL import Image
from keras.preprocessing.image import *
import scipy as sp
import keras.backend as K
import tensorflow as tf


def get_colormap():

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
    label_idx = 0
    for i in range(23):
        for j in range(23):
            if j not in valid_color_idx[i]:
                test_colormap[i,j,0] = 100
                test_colormap[i,j,1] = 0
                test_colormap[i,j,2] = 0
            else:
                ab_label_map[(int((i-11)*10), int((j-11)*10))] = label_idx
                label_idx+=1


    return ab_label_map
    #print(ab_label_map)

    lab_rgb_image = lab2rgb(test_colormap)
    #plt.imshow(lab_rgb_image)
    ##plt.show()


def get_weight(post_prob):
    LAMBDA = 0.25
    post_prob_smoothed = smooth(post_prob)
    weight = 1/( (1-LAMBDA)*post_prob_smoothed +LAMBDA/313 )
    factor = post_prob_smoothed.dot(weight)
    weight = weight/factor

    return weight

def smooth(post_prob):
    SIGMA = 5/10 # if use 5, the distribution is bad. Since the colorspace is scaled by 10.
    ab_label_map = get_colormap()

    """
    inv_ab_label_map = {}
    for k in ab_label_map:
        inv_ab_label_map[ab_label_map[k]] = k
    """
    # put the probability to 2d
    post_prob2d = np.zeros((23, 23))
    for i in range(23):
        for j in range(23):
            pos = ((i-11)*10,(j-11)*10)
            if pos in ab_label_map:
                label1d = ab_label_map[pos]
                post_prob2d[i, j] = post_prob[label1d]


    # smooth the probability
    post_prob2d_smoothed = sp.ndimage.gaussian_filter(post_prob2d,sigma = SIGMA)
    post_prob2d_smoothed /=sum(sum(post_prob2d_smoothed))

    # back to 1d
    post_prob_smoothed = np.zeros(313)
    cnt = 0
    for i in range(23):
        for j in range(23):
            pos = ((i-11)*10,(j-11)*10)
            if pos in ab_label_map:
                cnt += 1
                label1d = ab_label_map[pos]
                post_prob_smoothed[label1d] = post_prob2d_smoothed[i, j]
                #print (post_prob_smoothed)
                #print(cnt)
    """
       plt.imshow(post_prob2d_smoothed)
       test_colormap = np.ones((23, 23, 3))
       for i in range(23):
           for j in range(23):
               test_colormap[i, j,0] = post_prob2d_smoothed[i,j]*2000

       lab_rgb_image = lab2rgb(test_colormap)
       plt.imshow(lab_rgb_image)
    """

    return post_prob_smoothed #/sum(post_prob_smoothed)




def build_labels(): # build a table, mappingy_true to vector for cross entropy
    #load raw prior estimate and get the weight
    CLASSES = 313
    post_prob = np.load("./models/posterior_prob_flower.npy")
    post_prob = post_prob / np.sum(post_prob)
    weight = get_weight(post_prob)

    ab_label_map = get_colormap()
    inv_ab_label_map = {}
    for k in ab_label_map:
        v = ab_label_map[k]
        inv_ab_label_map[v] = k

    color_vec_table = np.zeros((CLASSES,CLASSES+1)) # +1 for the cross entropy.
    SIGMA = 0.2
    raw_nn_weight = np.exp(-1/(2*SIGMA**2))
    # DETAIL: abandon neighbours out bound. Rescale to normalize.
    for i in range(CLASSES):
        color_vec_table[i,i] = 1
        p1,p2 = inv_ab_label_map[i]
        for r,c in ((p1+10,p2),(p1-10,p2),(p1,p2+10),(p1,p2-10)):
            if (r,c) in ab_label_map:
                idx = ab_label_map[(r,c)]
                color_vec_table[i,idx] = raw_nn_weight
        color_vec_table[i,:] = color_vec_table[i,:]/np.sum(color_vec_table[i,:])
        color_vec_table[i, :] = color_vec_table[i, :]*weight[i]

    np.save("./models/color_vec_table_flower_0.2.npy",color_vec_table)



def pred_2_image_mode(pred,intensity):#pred: W,H,314. intensity: the grayscale. Return lab image
    CLASSES = pred.shape[-1] - 1
    pred = K.eval(pred).astype("float")
    pred = pred[:, :, :CLASSES]
    width = pred.shape[0]
    height = pred.shape[1]

    ab_label_map = get_colormap()
    inv_ab_label_map = {}
    for k in ab_label_map:
        v = ab_label_map[k]
        inv_ab_label_map[v] = k

    lab_image = np.zeros((width, height, 3))
    lab_image[:, :, 0] = intensity

    for i in range(width):
        for j in range(height):
            pixel_pred = pred[i,j,:]
            color_idx = np.argmax(pixel_pred)
            color = inv_ab_label_map[color_idx]
            lab_image[i,j,1:] = color

    return lab_image


def pred_2_image_anneal(pred, intensity):#pred:W*H*314, intensity: the grayscale. Return lab image
    CLASSES = pred.shape[-1]-1
    T = 0.38
    pred = K.eval(pred).astype("float")
    pred = pred[:,:,:CLASSES]
    width = pred.shape[0]
    height = pred.shape[1]
    ab_label_map = get_colormap()
    inv_ab_label_map = {}
    for k in ab_label_map:
        v = ab_label_map[k]
        inv_ab_label_map[v] = k

    lab_image = np.zeros((width,height,3))
    lab_image[:,:,0] = intensity

    colors = np.zeros((CLASSES,2))
    for i in range(CLASSES):
        colors[i,:] = np.array(inv_ab_label_map[i])

    for i in range(width):
        for j in range(height):
            pixel_pred = pred[i,j,:]
            tmp = np.exp(np.log(pixel_pred)/T)
            anneal_prob = tmp/np.sum(tmp)
            anneal_mean = anneal_prob.dot(colors)
            lab_image[i,j,1:] = anneal_mean

    return lab_image


def pixel_prob2color_5nn(pixel_pred,ab_label_map):
    SIGMA = 0.5
    colors_prob = np.zeros((23,23))
    for i in range(23):
        for j in range(23):
            pos = ((i-11)*10,(j-11)*10)
            if pos in ab_label_map:
                label1d = ab_label_map[pos]
                colors_prob[i, j] = pixel_pred[label1d]

    colors_prob = sp.ndimage.gaussian_filter(colors_prob, sigma=SIGMA)
    sorted_idx = np.dstack(np.unravel_index(np.argsort(colors_prob.ravel()), colors_prob.shape))[0]
    for i in range(1,23*23+1):
        p1,p2 = tuple((sorted_idx[-i,:]-11)*10)
        if (p1,p2) not in ab_label_map:
            continue
        raw_nn_weight = np.exp(-1 / (2 * SIGMA ** 2))
        raw_weight = np.array([1,raw_nn_weight,raw_nn_weight,raw_nn_weight,raw_nn_weight])
        prob_weight = np.array([ colors_prob[r//10+11,c//10+11] for (r,c) in ((p1,p2),(p1+10,p2),(p1-10,p2),(p1,p2+10),(p1,p2-10)) ])
        weight = raw_weight*prob_weight
        weight = weight/np.sum(weight)
        res = weight.dot(np.array(((p1,p2),(p1+10,p2),(p1-10,p2),(p1,p2+10),(p1,p2-10))))
        break

    return res



def pred_2_image_5nn(pred, intensity):#pred:W*H*314, intensity: the grayscale. Return lab image
    CLASSES = pred.shape[-1]-1
    pred = K.eval(pred).astype("float")
    pred = pred[:,:,:CLASSES]
    width = pred.shape[0]
    height = pred.shape[1]
    ab_label_map = get_colormap()
    inv_ab_label_map = {}
    for k in ab_label_map:
        v = ab_label_map[k]
        inv_ab_label_map[v] = k

    lab_image = np.zeros((width,height,3))
    lab_image[:,:,0] = intensity

    for i in range(width):
        for j in range(height):
            pixel_pred = pred[i,j,:]
            lab_image[i,j,1:] = pixel_prob2color_5nn(pixel_pred,ab_label_map)

    return lab_image


if __name__ == '__main__':
    build_labels()
