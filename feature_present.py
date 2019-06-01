# coding: utf-8
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import BCNN_VGG16

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = img_batch
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    print('num_pic',num_pic)
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')

    plt.savefig('feature_map.png')
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")


if __name__ == "__main__":
    base_model = BCNN_VGG16.build_model()
   # base_model = VGG16(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    base_model.load_weights('model/bcnn_keras_vgg16_finetune32_best.h5')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)


    img_path = 'data/test.jpg'
    img = image.load_img(img_path)
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block_pool_features = model.predict(x)
    print(block_pool_features.shape)

    feature = block_pool_features.reshape(block_pool_features.shape[1:])

    visualize_feature_map(feature)
