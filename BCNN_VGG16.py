from keras.utils.training_utils import multi_gpu_model
import numpy as np
import keras
import keras.metrics
from keras.utils import np_utils, conv_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation,normalization
from keras.optimizers import Adam
import math
from keras.layers import Lambda, Softmax, Dot,Permute
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
from PIL import ImageFile
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import pandas as pd
# the params of the input dataset.
# the width, height & channels of the images in the input TFRecords file.
#! -- coding: utf-8 --*--
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

# 设定 GPU 显存占用比例为 0.7
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
ktf.set_session(session )

ImageWidth = 224
ImageHeight = 224
ImageChannels = 3
# the number of categories.
CategoryNum = 1580

def outer_product(x):
    """
    calculate outer-products of 2 tensors

        args
            x
                list of 2 tensors
                , assuming each of which has shape = (size_minibatch, total_pixels, size_filter)
    """
    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1]

def signed_sqrt(x):
    """
    calculate element-wise signed square root

        args
            x
                a tensor
    """
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    """
    calculate L2-norm

        args
            x
                a tensor
    """
    return keras.backend.l2_normalize(x, axis=axis)
def build_model(
    ImageWidth = 224,
    ImageHeight = 224,
    ImageChannels = 3,
    name_optimizer="sgd",
    rate_learning=0.1,
    rate_decay_learning=0.0,
    rate_decay_weight=0.0,
    no_class = 1579,
    name_initializer="glorot_normal",
    name_activation_logits="softmax",
    name_loss="categorical_crossentropy",
    flg_debug=False,
    flg_Freeze = False,
    num_Freeze = -1
):
    print('------------model begins to building---------')
    model = keras.applications.VGG16(include_top=False,weights = 'imagenet',input_shape=(ImageWidth,ImageHeight,ImageChannels))
   # model = keras.applications.ResNet50(nclude_top=False,input_shape=(224,224,3))

    #The bilinear layer.
    # We combine the 2 identical convolution layers in our code. You can also combine 2 different convolution layers.
    # The bilinear layer is connected to the final output layer(the logits layer).
    ###
    ### bi-linear pooling
    ###

    # extract features from detector
    x_detector = model.layers[17].output
    shape_detector = model.layers[17].output_shape
    if flg_debug:
        print("shape_detector : {}".format(shape_detector))

    # extract features from extractor , same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector
    if flg_debug:
        print("shape_extractor : {}".format(shape_extractor))

    # rehape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape(
        [
            shape_detector[1] * shape_detector[2], shape_detector[-1]
        ]
    )(x_detector)
 #   x_detector = Permute((2, 1))(x_detector)
    if flg_debug:
        print("x_detector shape after rehsape ops : {}".format(x_detector.shape))

    x_extractor = keras.layers.Reshape(
        [
            shape_extractor[1] * shape_extractor[2], shape_extractor[-1]
        ]
    )(x_extractor)
   # x_extractor = Permute((2, 1))(x_extractor)
    if flg_debug:
        print("x_extractor shape after rehsape ops : {}".format(x_extractor.shape))

    # outer products of features, output shape=(minibatch_size, filter_size_detector*filter_size_extractor)

    x = keras.layers.Lambda(outer_product)(
        [x_detector, x_extractor]
    )
    if flg_debug:
        print("x shape after outer products ops : {}".format(x.shape))

    # rehape to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = keras.layers.Reshape([shape_detector[-1] * shape_extractor[-1]])(x)
    if flg_debug:
        print("x shape after rehsape ops : {}".format(x.shape))

    # signed square-root
    x = keras.layers.Lambda(signed_sqrt)(x)
    if flg_debug:
        print("x shape after signed-square-root ops : {}".format(x.shape))

    # L2 normalization
    x = keras.layers.Lambda(L2_norm)(x)
    if flg_debug:
        print("x shape after L2-Normalization ops : {}".format(x.shape))

    ###
    ### attach FC-Layer
    ###

    if name_initializer != None:
        name_initializer = eval(name_initializer + "()")



    x = keras.layers.Dense(
        units=no_class
        , kernel_regularizer=keras.regularizers.l2(rate_decay_weight)
        , kernel_initializer=name_initializer
    )(x)
    tensor_prediction = keras.layers.Activation(name_activation_logits)(x)

        ###
        ### compile model
        ###
    if flg_Freeze:
        for layer in model.layers[:num_Freeze]:
            layer.trainable = False
        for layer in model.layers[num_Freeze:]:
            layer.trainable = True
    else:
        for layer in model.layers:
            layer.trainable = False
    model_bilinear = keras.models.Model(
        inputs=model.input
        , outputs=[tensor_prediction]
    )
    model_bilinear.layers[-2].name =str(no_class)+ '_'+model_bilinear.layers[-2].name
    # fix pre-trained weights
    # for layer in model_detector.layers:
    #     layer.trainable = False

    # define optimizers
    opt_adam = keras.optimizers.adam(
        lr=rate_learning
        , decay=rate_decay_learning
    )
    opt_rms = keras.optimizers.RMSprop(
        lr=rate_learning
        , decay=rate_decay_learning
    )
    opt_sgd = keras.optimizers.SGD(
        lr=rate_learning
        , decay=rate_decay_learning
        , momentum=0.9
        , nesterov=False
    )
    optimizers = {
        "adam": opt_adam
        , "rmsprop": opt_rms
        , "sgd": opt_sgd
    }

    model_bilinear.compile(
        loss=name_loss
        , optimizer=optimizers[name_optimizer]
        , metrics=["categorical_accuracy"]

    )

    if flg_debug:
        model_bilinear.summary()
    print('model has already been built')
    return model_bilinear

def main(flag = 'train',flg_finetune = False,num = -1, lr = 0.1, flg_debug = True,classes_num = 32):
    TRAIN_DIR = './data/'+str(classes_num)+'/train'
    VALID_DIR = './data/'+str(classes_num)+'/train'
    TEST_DIR = './data/'+str(classes_num)+'/test'
    BATCH_SIZE = 8
    SIZE = (224, 224)
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])
    num_test_samples = sum([len(files) for r, d, files in os.walk(TEST_DIR)])
    num_train_steps = math.floor(num_train_samples *0.7/ BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples*0.3 / BATCH_SIZE)
    num_test_steps = math.floor(num_test_samples/BATCH_SIZE)
    print('num_train_steps: ', num_train_steps, 'num_valid_steps: ', num_valid_steps, 'num_test_steps: ',num_test_steps)
    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255., validation_split=0.3,horizontal_flip=True)

    tra_batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE,class_mode='categorical', shuffle=True,
                                      batch_size=BATCH_SIZE,subset='training')
    val_batches = gen.flow_from_directory(VALID_DIR, target_size=SIZE,class_mode='categorical', shuffle=True,
                                              batch_size=BATCH_SIZE,subset='validation')
    test_batches = gen.flow_from_directory(TEST_DIR, target_size=SIZE, class_mode='categorical',
                                          batch_size=BATCH_SIZE)

    CLASS =len( list(iter(tra_batches.class_indices)))
    MODE_NAME2 = 'bcnn_keras_vgg16_final'
    if(flg_finetune):
        MODE_NAME = 'bcnn_keras_vgg16_finetune' + str(CLASS)
    else:
        MODE_NAME = 'bcnn_keras_vgg16_'+str(CLASS)
    bcnn_model = build_model(flg_debug=flg_debug,no_class=CLASS,rate_learning=lr,rate_decay_weight=1e-9,name_optimizer='sgd',flg_Freeze=True,num_Freeze=num)
    #print(bcnn_model.to_json())
    if flag == 'train':
        # 载入权重
        print('Training----------------------------')
        print(len(bcnn_model.layers))
        if os.path.exists(MODE_NAME+'_best.h5'):
            print('---------previous mode is loading--------path:'+MODE_NAME+'_best.h5')
            bcnn_model.load_weights(filepath=MODE_NAME+'_best.h5')
        elif os.path.exists(MODE_NAME2+'.h5'):
            print('---------previous mode is loading--------path:' + MODE_NAME2 )
            bcnn_model.load_weights(filepath=MODE_NAME2+'.h5' ,by_name=True)

        checkpointer = ModelCheckpoint(MODE_NAME+'_best.h5', monitor='val_loss',verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=4)
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        history = bcnn_model.fit_generator(tra_batches, steps_per_epoch=num_train_steps, epochs=100,
                                           callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                                           validation_steps=num_valid_steps)
        bcnn_model.save('bcnn_keras_vgg16_final.h5')
        dp = pd.DataFrame(history.history)
        csv_path = 'history' + str(classes_num) + '.csv'
        if os.path.exists(csv_path):
            dp0 = pd.read_csv(csv_path)
            dp0 = dp0.append(dp)
            dp0.to_csv(csv_path, index=False)
        else:
            dp.to_csv(csv_path, index=False)
        # 绘制训练 & 验证的准确率值
        # plt.plot(history.history['categorical_accuracy'])
        # plt.plot(history.history['val_categorical_accuracy'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Valid'], loc='upper left')
        # plt.savefig('history.jpg')
    else :
        print('\nTesting ------------')  # 对测试集进行评估
        if os.path.exists(MODE_NAME + '_best.h5'):
            print('previous mode is loading------')
            bcnn_model.load_weights(filepath=MODE_NAME+'_best.h5')
        loss, accuracy=bcnn_model.evaluate_generator(test_batches, steps=num_test_steps)
        print('test loss: %.4f' % loss)
        print('test accuracy: %.4f' % accuracy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--train',action='store_true',default=False,
        help='train the model'
    )
    parser.add_argument(
        '--test', action='store_true',default=False,
        help='test the model'
    )
    parser.add_argument(
        '--num', type=int, default=-1,
        help='the num that last layer can be trained'
    )
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help='learning rate'
    )
    parser.add_argument(
        '--classes', type=int, default=32,
        help='the number of classes'
    )
    FLAGS = parser.parse_args()
    if FLAGS.train:

        main(flag='train', flg_finetune=True, num = FLAGS.num, lr=FLAGS.lr, classes_num=FLAGS.classes)
    else:

        main(flag='test', flg_finetune=True, classes_num=FLAGS.classes, flg_debug=True)