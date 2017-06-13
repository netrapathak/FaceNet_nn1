from __future__ import print_function
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.legacy.layers import MaxoutDense
from keras.models import Sequential, load_model
from keras import regularizers
from keras import backend as K
K.set_image_dim_ordering('th')
from collections import Counter
from PIL import Image
import numpy as np
import random
import os
import glob
import h5py

def lrn_layer(x, alpha=0.0005, beta=0.75, k=2):
    if K.image_dim_ordering == "th":
        _, f, r, c = x.shape
    else:
        _, r, c, f = x.shape
    squared = K.square(x)
    pooled = K.pool2d(squared, (5, 5), strides=(1, 1), padding="same", pool_mode="avg")
    if K.image_dim_ordering == "th":
        summed = K.sum(pooled, axis=1, keepdims=True)
        averaged = alpha * K.repeat_elements(summed, f, axis=1)
    else:
        summed = K.sum(pooled, axis=3, keepdims=True)
        averaged = alpha * K.repeat_elements(summed, f, axis=3)
    denom = K.pow(k + averaged, beta)
    return x / denom

def lrn_output_shape(input_shape):
    return input_shape

# %% Model:
def model(nb_pool, input_shape, data_format):

    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=7, strides=2, data_format=data_format, input_shape=input_shape, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=2, padding='same'))
    model.add(Lambda(lrn_layer,output_shape=lrn_output_shape))
    model.add(Convolution2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=2, padding='same'))
    model.add(Lambda(lrn_layer,output_shape=lrn_output_shape))
    model.add(Convolution2D(filters=192, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=2, padding='same'))
    model.add(Convolution2D(filters=384, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=1, strides=1, padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides=2, padding='same'))
    model.add(Flatten())
    model.add(MaxoutDense(32*128, 2))
    model.add(MaxoutDense(32*128,2))
    model.add(Dense(128, activity_regularizer=regularizers.l2(0.01)))

    print (model.summary())
    return model


# Assumption : Output will be a 2D array from FC7128 of dimension=(Batch_size,128)
def triplet_loss_facenet(y_true, y_pred):

    if (not all(K.int_shape(y_true)) == False):
        counts = Counter(y_true[:,0])
        class_idx = max(counts.iterkeys(), key=(lambda key: counts[key]))
        counts_class = counts.get(class_idx)
        positives = []
        negatives = []
        j = 0
        alpha = 0.2
        # create 2 lists containing indices of positives & negative images from batch
        for i in y_true[:,0]:
        	if i == class_idx:
        		positives.append(j)
        	else:
        		negatives.append(j)
        	j += 1
        positives = np.array(positives)

        # distance matrix for each positive pair
        dist_mat = np.zeros((len(positives),len(positives)))
        triplets = []
        pairs = []
        for i in range(dist_mat.shape[0]):
        	minimum = 1e+16
        	minimum_idx = [-1,-1]
        	for j in range(i+1, dist_mat.shape[1]):
        		if not i == j:
        			dist_mat[i,j] = dist_mat[j,i]=np.sum((y_pred[positives[i],:]-y_pred[positives[j],:])**2)
        			if dist_mat[i,j] < minimum: # minimum
        				minimum = dist_mat[i,j]
        				minimum_idx = [i,j]
        	if not minimum_idx == [-1,-1]:
        		triplets.append([positives[minimum_idx[0]],positives[minimum_idx[1]]])
        		triplets.append([positives[minimum_idx[1]],positives[minimum_idx[0]]])
        		pairs.append([minimum_idx[0],minimum_idx[1]])
        		pairs.append([minimum_idx[1],minimum_idx[0]])

        size = len(triplets)
        loss = 0
        for i in range(size):
        	minimum = 1e+16
        	minimum_idx2 = -1
        	for j in range(len(negatives)):
        		dst_tmp = np.sum((y_pred[triplets[i][0],:]-y_pred[negatives[j],:])**2)
        		if dst_tmp < minimum and dist_mat[pairs[i][0]][pairs[i][1]] < dst_tmp: # semi-hard negative
        			minimum = dst_tmp
        			minimum_idx2 = j
        	if not minimum_idx == -1:
        		triplets[i].append(negatives[minimum_idx2])
        	if minimum < 1e+16:
        		loss += (alpha + dist_mat[pairs[i][0]][pairs[i][1]] - minimum)
        else:
            loss = 0

        return loss


def loadModel(modelFilePath):
    loaded_model = load_model(modelFilePath)
    print ("Loaded model from disk")
    loaded_model.compile(loss=triplet_loss, optimizer='Adagrad', metrics=["accuracy"])
    return loaded_model

def readImage(imgPath, img_height, img_width, img_channels=3):
    im = Image.open(imgPath)
    img = np.array(im.resize((img_height,img_width)))
    if (len(img.flatten()) < (img_height*img_width*img_channels)):
        print ('Incorrect image shape')
    return img.flatten()

def createData(train_data_dir, img_height, img_width, batch_size):
    x = []
    y = []
    imgList = []
    classes = os.listdir(train_data_dir)
    for index in range(len(classes)):
        imgVectorList = []
        folderPath = os.path.join(train_data_dir,classes[index])
        images = glob.glob(folderPath+'/*.jpg')
        for i in range(len(images)):
            imgArr = readImage(images[i], img_height, img_width)
            imgVectorList.append(imgArr)
        imgList.append([index,imgVectorList])

    for i in range(400): # 400 batches
        posClass = random.randint(0,(len(classes)-1))
        x.extend(random.sample(imgList[posClass][1], 40)) #positive imgList
        y.extend([[imgList[posClass][0] for i in range(128)] for i in range(40)])

        negIdx = []
        counter = 0
        while (counter < (batch_size-40)):
            i = random.randint(0, (len(imgList)-1))
            if (i != imgList[posClass][0]):
                nb = len(imgList[i][1])-1
                x.append(imgList[i][1][random.randint(0,nb)])
                y.append([imgList[i][0] for i in range(128)])
                counter += 1

    x = np.array(x)
    y = np.array(y)
    return x, y
