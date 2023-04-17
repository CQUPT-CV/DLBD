import glob
import math
import os
import random
import torch
import numpy as np
import pandas as pd

from cifar10_pair import CIFAR10SimPair
from DLBD import DLBD
from UWTXent import UWTXent
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# hyper-parameter
bit = 32
use_cuda = True
lr = 0.001
epoch_num = 200
dataset_path = "/asset/dataset/cifar10/"


# epoch代表一次完整训练集完成训练的过程
def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    # 每次取出数量为batch size的数据
    for i, (input1, input2, y) in enumerate(train_loader):
        if use_cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            y = y.cuda()
        bin_output1 = model(input1)
        bin_output2 = model(input2)
        loss = criterion(bin_output1, bin_output2)
        # print("i:", i, " loss:", torch.mean(loss))

        # 每一轮batch需要设置optimizer.zero_grad
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        # print("i:", i, " cur loss:", np.mean(train_loss))
    return np.mean(train_loss)

def bin_tensor2df(tensor_2d):
    size = tensor_2d.shape
    result = pd.DataFrame()
    for i in range(size[0]):
        list_tensor = []
        for j in range(size[1]):
            # a = tensor_2d[i][j].item()
            list_tensor.append('%.3f' % tensor_2d[i][j].item())
        result = result.append([list_tensor], ignore_index=True)
        # result = pd.concat([result, pd.Series(list_tensor)], axis=1)
    return result

def test_cuda():
    # 返回当前设备索引
    print(torch.cuda.current_device())
    # 返回GPU的数量
    print(torch.cuda.device_count())
    # 返回gpu名字，设备索引默认从0开始
    print(torch.cuda.get_device_name(0))
    # cuda是否可用
    print(torch.cuda.is_available())


def adjust_learning_rate(optimizer, epoch):
    cur_lr = lr
    cur_lr *= 0.5 * (1. + math.cos(math.pi * epoch / epoch_num))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


if __name__ == '__main__':

    model = DLBD(bit)
    
    criterion = UWTXent(t=8, dim=bit, divide_num=2)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-6)
    best_loss = 10000.0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    
    transforms_c = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    if use_cuda:
        model = model.cuda()

    train_data = CIFAR10SimPair(dataset_path, train=True, download=False, transform=transforms_c)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=256,
                                               shuffle=True)

    mAPs = []
    for epoch in range(epoch_num):
        adjust_learning_rate(optimizer, epoch)
        train_loss = train(train_loader, model, criterion, optimizer)
        print('Epoch: {0}, Train loss: {1}'.format(epoch, train_loss))
        scheduler.step()
        torch.save(model.state_dict(), '/model_DLBD_cifar10_{0}e.pth'.format(epoch))