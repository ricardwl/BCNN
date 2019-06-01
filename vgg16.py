import numpy as np
import keras
import keras.metrics
from keras import layers
from keras.utils import np_utils, conv_utils
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation,normalization
from keras import  optimizers
import math
from keras.layers import Lambda, Softmax, Dot,Permute
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import argparse
import pickle
import pandas as pd
# the params of the input dataset.
# the width, height & channels of the images in the input TFRecords file.
ImageWidth = 224
ImageHeight = 224
ImageChannels = 3
# the number of categories.
def build_model(
    ImageWidth = 224,
    ImageHeight = 224,
    ImageChannels = 3,
    name_optimizer="sgd",
    rate_learning=0.01,
    rate_decay_learning=0.0,
    rate_decay_weight=1e-9,
    no_class = 1579,
    name_initializer="glorot_normal",
    name_activation_logits="softmax",
    name_loss="categorical_crossentropy",
    flg_debug=False,
    flg_Freeze = False,
    num_Freeze = 1
):
    print('------------model begins to building---------')
    base_model = keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(ImageWidth,ImageHeight,ImageChannels))


    out = base_model.layers[-1].output
    out = layers.Flatten()(out)
    out = layers.Dense(1024, activation='relu')(out)
    # 因为前面输出的dense feature太多了，我们这里加入dropout layer来防止过拟合
    #out = layers.Dropout(0.5)(out)
    out = layers.Dense(512, activation='relu')(out)
    #out = layers.Dropout(0.5)(out)
    out = layers.Dense(no_class, activation=name_activation_logits)(out)
    tuneModel = Model(inputs=base_model.input, outputs=out)
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

    for layer in tuneModel.layers[:num_Freeze]:  # freeze the base model only use it as feature extractors
        layer.trainable = False
    if flg_debug:
        print(tuneModel.summary())
    tuneModel.compile(loss='categorical_crossentropy', optimizer=optimizers[name_optimizer],
                      metrics=['categorical_accuracy'])
    return tuneModel


def main(flag = 'train',flg_finetune = False,num = -1, lr = 0.01, flg_debug = True,classes_num = 32):
    TRAIN_DIR = './data/'+str(classes_num)+'/train'
    VALID_DIR = './data/'+str(classes_num)+'/train'
    TEST_DIR = './data/'+str(classes_num)+'/test'
    BATCH_SIZE = 16
    SIZE = (224, 224)
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])
    num_test_samples = sum([len(files) for r, d, files in os.walk(TEST_DIR)])
    num_train_steps = math.floor(num_train_samples * 0.7 / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples * 0.3 / BATCH_SIZE)
    num_test_steps = math.floor(num_test_samples / BATCH_SIZE)
    print('num_train_steps:', num_train_steps, 'num_valid_steps:', num_valid_steps, 'num_test_steps: ', num_test_steps)
    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255., validation_split=0.3, horizontal_flip=True)
    gen2 = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.)
    tra_batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                          batch_size=BATCH_SIZE, subset='training')
    val_batches = gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                          batch_size=BATCH_SIZE, subset='validation')
    test_batches = gen2.flow_from_directory(TEST_DIR, target_size=SIZE, class_mode='categorical',
                                           batch_size=BATCH_SIZE)

    CLASS =len( list(iter(tra_batches.class_indices)))


    MODE_NAME2 = 'VGG16_finall'
    if (flg_finetune):
        MODE_NAME = 'VGG16_finetune' + str(CLASS)
    else:
        MODE_NAME = 'VGG16_' + str(CLASS)
    bcnn_model = build_model(flg_debug=flg_debug,no_class=CLASS,rate_learning=lr,name_optimizer='sgd',flg_Freeze=True,num_Freeze=num)
  #  print(bcnn_model.to_json())
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

        #bcnn_model.load_weights(filepath=MODE_NAME2 ,by_name=True)

        checkpointer = ModelCheckpoint(MODE_NAME+'_best.h5', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=4)
        history = bcnn_model.fit_generator(tra_batches, steps_per_epoch=num_train_steps, epochs=100,
                                           callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                                           validation_steps=num_valid_steps)
        bcnn_model.save('VGG16_final.h5')
        dp = pd.DataFrame(history.history)
        csv_path = 'history' + str(classes_num) + '.csv'
        if os.path.exists(csv_path):
            dp0 = pd.read_csv(csv_path)
            dp0 = dp0.append(dp)
            dp0.to_csv(csv_path, index=False)
        else:
            dp.to_csv(csv_path, index=False)


    else:
        print('\nTesting ------------')  # 对测试集进行评估
        if os.path.exists(MODE_NAME + '_best.h5'):
            print('previous model is loading------')
            bcnn_model.load_weights(filepath=MODE_NAME + '_best.h5')
        loss, accuracy = bcnn_model.evaluate_generator(test_batches, steps=num_test_steps)
        print('test loss: %.4f' % loss)
        print('test accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--train', action='store_true', default=False,
        help='train the model'
    )
    parser.add_argument(
        '--test', action='store_true', default=False,
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

        main(flag='train', flg_finetune=True, num=FLAGS.num, lr=FLAGS.lr, classes_num=FLAGS.classes)
    else:

        main(flag='test', flg_finetune=True, classes_num=FLAGS.classes, flg_debug=False)