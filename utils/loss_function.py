from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import numpy as np

#load in the color_vec_table

weight_path = "/home/zc2388/segmentation/Keras-Colorization/models/color_vec_table_cat_0.2.npy"
COLOR_VEC_TABLE = np.load(weight_path) #("/home/zc2388/segmentation/Keras-FCN/models/color_vec_table_cat_0.2.npy")
COLOR_VEC_TABLE = np.vstack((COLOR_VEC_TABLE,np.zeros((1,314))))
COLOR_VEC_TABLE = np.vstack((COLOR_VEC_TABLE,np.zeros((1,314))))
COLOR_VEC_TABLE = K.constant(COLOR_VEC_TABLE) # 314*314


def softmax_sparse_crossentropy_ignoring_last_label_weighted_tensor(y_true, y_pred):# y_true B,W,H,1
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.reshape(y_true, (-1, K.int_shape(y_true)[-1]))
    y_true = K.squeeze(y_true,1)
    y_true = tf.dtypes.cast(y_true,tf.int32) # force the type to be int, maybe not needed?
    y_true = tf.gather(COLOR_VEC_TABLE,y_true)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)


def L2_ab_regression_loss(ab_pred, ab_true):
    total_loss = 0

    mse_error_stage1 = K.variable(0.005) * K.mean(K.square(ab_pred - ab_true), axis=-1)
    mse_error_vector = K.reshape(mse_error_stage1, (-1, 1))
    total_loss = K.mean(mse_error_vector, axis=0)
    return total_loss

def L1_ab_regression_loss(ab_pred, ab_true):
    total_loss = 0

    mse_error_stage1 = K.mean(K.abs(ab_pred - ab_true), axis=-1)
    mse_error_vector = K.reshape(mse_error_stage1, (-1, 1))
    total_loss = K.mean(mse_error_vector, axis=0)
    return total_loss

def L12_ab_regression_loss(ab_pred, ab_true):
    total_loss = 0

    abs_error = K.mean(K.abs(ab_pred - ab_true), axis=-1)
    mse_error = K.variable(0.1) * K.mean(K.square(ab_pred - ab_true), axis=-1)

    smaller_one = K.minimum(abs_error, mse_error)
    mse_error_vector = K.reshape(smaller_one, (-1, 1))
    total_loss = K.mean(mse_error_vector, axis=0)
    return total_loss

