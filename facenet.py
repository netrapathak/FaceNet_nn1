# for GPU:
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python facenet.py

from __future__ import print_function
from utility import model, triplet_loss_facenet, createData
from keras.optimizers import SGD,Adagrad
from keras.utils import np_utils
from keras import backend as K
import numpy as np


if __name__ == "__main__":

    # In the block below, parameters have to be defined
    #---------------------------------------------------------------------------
    # input image dimensions:
    img_height, img_width = 20,20
    epochs = 10
    batch_size = 200
    train_data_dir = './FacenetDataset/trainFaces'
    # validation_data_dir = ''

    #---------------------------------------------------------------------------
    # Fine-Tuning parameters:

    # number of channels
    img_channels = 3
    # size of pooling area for max pooling
    nb_pool = 3
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    #---------------------------------------------------------------------------
    X_train, Y_train = createData(train_data_dir, img_height, img_width, batch_size)

    print ('X_train shape: ', X_train.shape)
    print ('Y_train shape: ', Y_train.shape)

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], img_channels, img_height, img_width)
        input_shape = (img_channels, img_height, img_width)
        data_format = "channels_first"
    else:
        X_train = X_train.reshape(X_train.shape[0], img_height, img_width, img_channels)
        input_shape = (img_height, img_width, img_channels)
        data_format = "channels_last"

    X_train = X_train.astype('float32')
    Y_train = Y_train.astype('float32')
    X_train /= 255
    #---------------------------------------------------------------------------

    model = model(nb_pool, input_shape, data_format)

    # load previous model
    # model = load_model('')

    model.compile(loss=triplet_loss_facenet, optimizer='Adagrad', metrics=["accuracy"])
    model.fit(x=X_train, y=Y_train, batch_size=batch_size, verbose=1, epochs=epochs)
    #
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open('model.json', "w") as json_file:
    #     json_file.write(model_json)

#-------------------------------------------------------------------------------
# for SGD as optimizer
#----------------------
# lrate = 0.001
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
