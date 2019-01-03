UNI: zc2388 and zz2559

This repo is a modified version of Keras-FCN repo, which is a Keras tool for image segmentation. link: https://github.com/aurora95/Keras-FCN

This reference github repo provide a Keras version of resnet, a Keras version bilinear upsample layer and a cross-entropy loss function for image segmentation. Since Keras does not have those layers for image segmentation, so we reused the well-designed tool to do image segmentation pretrain. Here instead of just doing image segmentation, several tools are designed to finish image colorization task.

# util classes and functions:

utils/Coloriazation_generator.py: This is a data generator for Keras training using regression loss. We read source RGB image from a given path, use skimage library to format it into Lab color space. Then we'll use only L channel to reconstruct grayscale image as input for our network. If we are using regression loss, then the desired output is just ab channel. For data augmentation part we followed what Keras-FCN did. The fill value is fixed to be zero for both input image and label.

utils/loss_function.py: Several loss layers are added here to do label rebalance. Regression losses are also defined here. softmax_sparse_crossentropy_ignoring_last_label_weighted_tensor(y_true, y_pred) is the loss with best performance.

compute_color_posterior.py: Running this script is the first step needed to be done for Colorization as Segmentation. This script will scan the whole training dataset and get a prior prob-distribution (not posterior!). This matrix will later be used in computing weight for each class. To run this script, training dir should be given in this script, and then run 'python compute_color_posterior.py'. Since we tried different classes, we specify the name of output file. Output will be saved in 'models' directory.

get_color_class.py: This tool takes our dataset and color map, then produce label file (.npy format) for each image. The label map will later be used in Colorization as Segmentation task. Just run 'python get_color_class.py'. data dir, train file path and output dir can be edited inside this file. (Need to be given correctly)

get_color_util.py: This util function helps us get the color distribution. We'll go through training dataset and collect color distribution.(by calling compute_color_posterior.py) Using this matrix, we are able to calculate weight for each label. The parameter is then saved to a matrix of shape 313 by 314. The weight matrix will later be used in our weighted cross-entropy loss function. Output saved in models/ dir.

LR_SGD.py: this is a modified version of SGD optimizer. Since the first half part of our model is already well trained on ImageNet dataset, we'll make the learning rate to be smaller than randomly init params.
reference: https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/

# Training scripts:

train.py: This is the original training script for image segmentation task from https://github.com/aurora95/Keras-FCN . Given training dataset, we get a image segmentation model.

train_colorization_as_regression.py: Use Regression loss to train image colorization task. Just adjust param in this script and run python train_colorization_as_regression.py

colorization_as_segmentation_train.py: Use Segmentation method to train image colorization task. This script use normal cross-entropy loss without rebalance them. So you should not expect to get some convincing result. For running just adjust param in this script and run python 'colorization_as_segmentation_train.py'

colorization_as_segmentation_rebalance_train.py: Use Segmentation method to train image colorization task. Since we might have multiple classes and you might want to train model for only one class. In this case our weight matrix will be different. So the path to weight matrix is hard-coded inside utils/loss.py, in order to run properly this path also need to be changed. For running just adjust param in this script and run 'python colorization_as_segmentation_rebalance_train.py'
