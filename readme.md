UNI: zc2388 and zz2559

This repo is a modified version of Keras-FCN repo, which is a Keras tool for image segmentation. link: https://github.com/aurora95/Keras-FCN

This reference github repo provide a Keras version of resnet, a Keras version bilinear upsample layer and a cross-entropy loss function for image segmentation. Since Keras does not have those layers for image segmentation, so we reused the well-designed tool to do image segmentation pretrain.

utils/Coloriazation_generator.py: This is a data generator for Keras training using regression loss. We read source RGB image from a given path, use skimage library to format it into Lab color space. Then we'll use only L channel to reconstruct grayscale image as input for our network. If we are using regression loss, then the desired output is just ab channel. For data augmentation part we followed what Keras-FCN did. The fill value is fixed to be zero for both input image and label.

utils/loss.py: Several loss layers are added here to do label rebalance. Regression losses are also defined here.

get_color_util.py: This util function helps us get the color distribution. We'll go through training dataset and collect color distribution. Using this matrix, we are able to calculate weight for each label. The parameter is then saved to a matrix of shape 314 by 314. The weight matrix will later be used in our weighted cross-entropy loss function

LR_SGD.py: this is a modified version of SGD optimizer. Since the first half part of our model is already well trained on ImageNet dataset, we'll make the learning rate to be smaller than randomly init params.

train.py: Use Regression loss to train image colorization task. Just adjust param in this script and run python train.py

colorization_as_segmentation_train.py: Use Segmentation method to train image colorization task. Just adjust param in this script and run python colorization_as_segmentation_train.py
