import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import setting
import random
from util import util
import cv2

# torch.cuda.set_device(gpu_id)#使用GPU
# learning_rate = 0.0001
# input_layer_num = 32
# output_layer_num = 16


# 数据集的设置*****************************************************************************************************************
root = [os.getcwd()+'\\twitter_train.txt',os.getcwd()+'\\twitter_VP_train.txt',os.getcwd()+'\\twitter_SecondLady_train.txt'] # 调用图像
input_layer_num = setting.input_layer_num
output_layer_num = setting.output_layer_num



# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class RandSeqDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,sequence_len,train_histogram=True,img_database_location="D:\\UCID_color"):  # 初始化一些需要传入的参数
        super(RandSeqDataset, self).__init__()  # 对继承自父类的属性进行初始化
        self.len = sequence_len
        self.train_histogram = train_histogram
        self.img_database_location = img_database_location
        if train_histogram:
            #读取目标文件夹得所有图像文件地址
            self.imgs = os.listdir(img_database_location)
        # self.transform = transform
        # self.target_transform = target_transform
        # self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        # Generate Random Sequences
        if not self.train_histogram:
            random_int_list = np.array([0.0] * self.len, dtype=float)
            for i in range(self.len):
                random_int_list[i] = random.random()
            return torch.tensor(random_int_list, dtype=torch.float32), torch.tensor(
                setting.target_entropy + 0.4 * random.random(), dtype=torch.float32)  # 目标entropy

        else:
            # 如果想训练Histogram Layer，就不能一直使用随机序列，不然输出结果都太接近了（非常均匀的分布）
            img_pixels = self.handle_img(self.img_database_location+"\\"+self.imgs[index])

            histogram = util.get_histogram(img_pixels)
            # hist_float = [float(histogram[i])/sum(histogram) for i in range(len(histogram))]
            # entropy = util.entropy_calculate(histogram)
            return torch.tensor(img_pixels, dtype=torch.float32), torch.tensor(histogram,
                                                                                    dtype=torch.float32)  # 长度为默认256bin的histogram

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return 1000

    def handle_img(self,img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(128,128))
        img = img.flatten()
        # img = [float(img[i])/256 for i in range(len(self.imgs))]
        return img/255.0


# 根据自己定义的那个MyDataset来创建数据集！注意是数据集！而不是loader迭代器
# *********************************************数据集读取完毕********************************************************************
# 图像的初始化操作
if __name__ == '__main__':
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop((227, 227)),
    #     transforms.ToTensor(),
    # ])
    # text_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop((227, 227)),
    #     transforms.ToTensor(),
    # ])

    # 数据集加载方式设置
    train_data = RandSeqDataset(sequence_len=setting.carrier_weight_num)
    # test_data = MyDataset(txt=root + '_test.txt', transform=transforms.ToTensor())
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data)
        # print(batch_idx) nums:32*31

    # test_loader = DataLoader(mydataset=test_data, batch_size=6, shuffle=False, num_workers=4)
    print('num_of_trainData:', len(train_data))
    # print('num_of_testData:', len(test_data))
