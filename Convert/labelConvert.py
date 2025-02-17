"""
make label
@author: ZhangJian
"""

from skimage.io import imread, imsave
import skimage
import numpy as np
import os


def label(img_path, new_path, file_name="label2.tif"):
    img = imread(img_path)
    #print(img)
    print(img.shape)
    width = img.shape[0]
    height = img.shape[1]
    print(type(img[0][0]))
    new_image = np.random.randint(0, 256, size=[width, height], dtype=np.uint8)
    print(np.array([0, 255, 0]))


    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if all(np.array(img[i][j]) == np.array([255, 0, 0])):  # 红色，建成区
                #new_image[i][j] = 100
                new_image[i][j] = 1
            elif all(np.array(img[i][j]) == np.array([0, 255, 0])):  # 绿色，农用地
                #new_image[i][j] = 50
                new_image[i][j] = 2
            elif all(np.array(img[i][j]) == np.array([0, 255, 255])):  # 天蓝色，林地
                #new_image[i][j] = 150
                new_image[i][j] = 3
            elif all(np.array(img[i][j]) == np.array([255, 255, 0])):  # 黄色，草地
                #new_image[i][j] = 200
                new_image[i][j] = 4
            elif all(np.array(img[i][j]) == np.array([0, 0, 255])):  # 蓝色，水系
                #new_image[i][j] = 250
                new_image[i][j] = 5
            else:
                new_image[i][j] = 0
            pass

    imsave(new_path+file_name, new_image)

    #skimage.io.imshow(new_image)
    #skimage.io.show()


if __name__ == "__main__":
    # 读取文件夹下的所有图片，并进行处理，存储到相应文件夹下
    files = os.listdir("H:\深度学习二期\武大公开数据\label_5classes")
    for file in files:
        print("--------------%s" % file)
        label("H:\\深度学习二期\\武大公开数据\\label_5classes\\"+file, "H:\\深度学习二期\武大公开数据\\mylabel\\", file)