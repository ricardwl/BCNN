import shutil
import os
import random


def deletenum():
    filename = '../data/all/train'
    ls = os.listdir(filename)
    for dir in ls:
        num = len(os.listdir(filename+'/'+dir))
        if num <= 5:
            print('delete: ', dir, num)
            shutil.rmtree(filename+'/'+dir)


def split2dir():
    src_path = '../data/image_complete'
    dst_path = '../data/all/train'
    imgs = os.listdir(src_path)
    classes = []
    i = 0
    for img in imgs:
        i = i + 1
        if i % 1000 == 0:
            print(i)
        brand = img.split(' ', maxsplit=1)[0]
        dir_path = dst_path + '/' + brand
        img_src = src_path + '/' + img
        img_dst = dir_path + '/' + img

        if not os.path.exists(dir_path):
            # make the new class dir
            os.makedirs(dir_path)
            print('create: '+brand)
        shutil.move(img_src, img_dst)
        #print(img_src,'--->',img_dst)


def split_test():
    # move train data to test
    train_path = '../data/all/train'
    test_path = '../data/all/test'
    rate = 0.2
    dirs = os.listdir(train_path)
    j = 0
    for dir in dirs:
        j = j + 1
        if j % 100 == 0:
            print(j)
        if not os.path.exists(test_path+'/'+dir):
            os.makedirs(test_path+'/'+dir)
        imgs = os.listdir(train_path+'/'+dir)
        num = len(imgs)
        i = 0
        for img in imgs:
            img_src = train_path+'/'+dir+'/'+img
            img_dst = test_path+'/'+dir+'/'+img
            shutil.move(img_src, img_dst)
            i = i + 1
            if i > rate*num:
                break
def check():
    train_path = '../data/all/train'
    test_path = '../data/all/test'
    dirs = os.listdir(test_path)
    for dir in dirs:
        imgs = os.listdir(test_path+'/'+dir)
        for img in imgs:
            size = os.path.getsize(test_path+'/'+dir+'/'+img)
            if size == 0:
                print(img)
    for dir in dirs:
        imgs = os.listdir(train_path+'/'+dir)
        for img in imgs:
            size = os.path.getsize(train_path+'/'+dir+'/'+img)
            if size == 0:
                print(img)
    print('over!')

def gen_subdataset():
    # generate the subdataset
    classes_num = 32
    train_path = '../data/all/train'
    test_path = '../data/all/test'
    sub_classes_path = '../data/'+str(classes_num)
    if os.path.exists(sub_classes_path):
        shutil.rmtree(sub_classes_path)
    os.makedirs(sub_classes_path)
    sub_train_path = '../data/'+str(classes_num)+'/train'
    if not os.path.exists(sub_train_path):
        print('create: ',sub_train_path)
        os.makedirs(sub_train_path)
    sub_test_path = '../data/'+str(classes_num)+'/test'
    if not os.path.exists(sub_test_path):
        print('create: ', sub_test_path)
        os.makedirs(sub_test_path)
    classes = os.listdir(train_path)
    sub_calsses = random.sample(classes, classes_num)
    # copy the train data to the sub_train
    for dir in sub_calsses:
        dir_src = train_path+'/'+dir
        dir_dst = sub_train_path+'/'+dir
        print(dir_src, '-->', dir_dst)
        shutil.copytree(dir_src, dir_dst)
    # copy the test data to the sub_test
    for dir in sub_calsses:
        dir_src = test_path+'/'+dir
        dir_dst = sub_test_path+'/'+dir
        print(dir_src, '-->', dir_dst)
        shutil.copytree(dir_src, dir_dst)


if __name__ == '__main__':
    ob = 3
    if ob == 0:
        split2dir()
    elif ob == 1:
        deletenum()
    elif ob == 2:
        split_test()
    elif ob == 3:
        gen_subdataset()
    elif ob == 4:
        check()