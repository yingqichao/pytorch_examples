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
    def __init__(self,sequence_len):  # 初始化一些需要传入的参数
        super(RandSeqDataset, self).__init__()  # 对继承自父类的属性进行初始化
        self.len = sequence_len
        # self.imgs = imgs
        # self.transform = transform
        # self.target_transform = target_transform
        # self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        # Generate Random Sequences
        random_int_list = [0] * self.len
        for i in range(self.len):
            random_int_list[i] = random.random()
        return torch.tensor(random_int_list), torch.tensor(setting.target_entropy+0.4*random.random()) #不需要标签

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return 1000


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
