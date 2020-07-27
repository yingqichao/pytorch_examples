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

# torch.cuda.set_device(gpu_id)#使用GPU
learning_rate = 0.0001
# input_layer_num = 32
# output_layer_num = 16


# 数据集的设置*****************************************************************************************************************
root = [os.getcwd()+'\\twitter_train.txt',os.getcwd()+'\\twitter_VP_train.txt',os.getcwd()+'\\twitter_SecondLady_train.txt'] # 调用图像
input_layer_num = setting.input_layer_num
output_layer_num = setting.output_layer_num

# 定义读取图片文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


def twitter_loader_compress(strs='test'):
    # ASCII可用：32' '~126'~'（含）
    ret, num = [0.0] * input_layer_num, [0] * input_layer_num
    for i in range(len(strs)):
        ret[i%input_layer_num] += ord(strs[i])-ord(' ')
        num[i%input_layer_num] += 1
    ret = [ret[i]/num[i]/(ord('~')-ord(' ')) for i in range(input_layer_num)]
    return ret


def twitter_loader_first(strs='test'):
    # ASCII可用：32' '~126'~'（含）
    ret, num = [0.0] * input_layer_num, [0] * input_layer_num
    for i in range(input_layer_num):
        ret[i] += (ord(strs[i]) - ord(' '))/(ord('~')-ord(' '))
    return ret


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class TwitterDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txts=root, transform=None, target_transform=None, loader=twitter_loader_first):  # 初始化一些需要传入的参数
        super(TwitterDataset, self).__init__()  # 对继承自父类的属性进行初始化
        imgs, num_twitter = [], 0
        for txt in txts:
            fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
            for line in fh:  # 迭代该列表#按行循环txt文本中的内
                line = line.strip('\n')
                line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
                # words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
                if len(line) < input_layer_num:
                    continue
                imgs.append((line, num_twitter % output_layer_num))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
                num_twitter += 1
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片,这里fn是string，需要处理
        # if self.transform is not None:
        #     img = self.transform(img)  # 数据标签转换为Tensor
        # tran = transforms.ToTensor()
        # return tran(img), tran(label)  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return torch.tensor(img), torch.tensor(label)

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


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
    train_data = TwitterDataset(transform=transforms.ToTensor())
    # test_data = MyDataset(txt=root + '_test.txt', transform=transforms.ToTensor())
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=True, num_workers=4)
    # test_loader = DataLoader(mydataset=test_data, batch_size=6, shuffle=False, num_workers=4)
    print('num_of_trainData:', len(train_data))
    # print('num_of_testData:', len(test_data))
