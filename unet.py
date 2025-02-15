import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config



class Unet(object): #object新式类
    _defaults = { #受保护的属性，存储默认类的配置或参数
        "model_path"    : 'logs/best_epoch_weights.pth',    #logs日志下的权值文件
        "num_classes"   : 7+1,  #区分的类别 background = 0，
        "backbone"      : "vgg",    #所使用的主干网络：！vgg调为
        "input_shape"   : [512, 512],
        "mix_type"      : 0,    #检测结果的可视化方式 0：原图与生产图混合，1：仅生产图，2：扣去背景，仅保留原图目标
        "cuda"          : True, #将计算任务分配到 GPU 的数千个核心上，显著加速计算密集型任务
    }

    def __init__(self, **kwargs):   #keyword arguments任意数量关键字参数，打包为字典
        self.__dict__.update(self._defaults)    #self实例，defaults默认属性更新至dict属性字典中
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hls_to_rgb(*x), hsv_tuples))  #将hsv转换为rgb 【hue, saturation, value】
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors)) #hsv均匀分布，rgb图像显示+系统处理

        self.generate()
        show_config(**self._defaults)

    #加载一个U-Net模型及其权重
    def generate(self, onnx=False): #onnx/open neutral network exchange标准的模型表示方式，布尔型
        self.net = unet(num_classes = self.num_classes, backbone = self.backbone)   #unet网络定义
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       #设备gpu
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))  #加载预训练的模型权重到当前模型中
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        image = cvtColor(image) #转换图像为RGB

        old_img = copy.deepcopy(image) #备份原始输入图像
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))    #增加灰条，保持比例不变
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)),(2, 0, 1)), 0) #axis单个维度/轴；axes多个复数；转变为深度学习输入格式，将单张照片包装为一个batch

        with torch.no_grad():   #前向传播中禁用梯度
            images = torch.from_numpy(image_data)   #通过numpy将数据转化为pytorch张量（多维数组/基本数据结构）
            if self.cuda:
                images = images.cuda()  #是否支持GPU运算

            pr = self.net(images)[0]    #前向传播，通过模型；取输出的第一维【prediction】

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),    #图像裁剪，截取灰条
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)   #resize利用插值法（用已知推未知）放缩图像

            pr = pr.argmax(axis=-1)     #根据索引的概率输出像素点的种类

            if count:   ###
                classes_nums = np.zeros([self.num_classes])     #初始化num_classes数组

                total_points_num = orininal_h * orininal_w      #计算整张图片的像素点

                print('-' * 63)
                print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
                print('-' * 63)

                for i in range(self.num_classes):
                    num = np.sum(pr == i)
                    ratio = num / total_points_num * 100    #计算每类占比
                    if num > 0:
                        print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                        print('-' * 63)
                    classes_nums[i] = num
                print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])   #根据预测结果生成一张彩色分割图像

            image = Image.fromarray(np.uint8(seg_img))

            image = Image.blend(old_img, image, 0.7)    #融合新与原图

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))

    def get_FPS(self, image, test_interval):
        image = cvtColor(image)

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                pr = self.net(images)[0]

                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time