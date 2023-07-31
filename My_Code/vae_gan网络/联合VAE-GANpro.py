#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置画图的尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
img_dir = r"E:\data_all\MNIST"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def show_images(images):  # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


def deprocess_img(x):
    return (x * 0.5) + 0.5


# 数据处理
im_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5])  # 标准化
])
train_set = MNIST(img_dir, download=True, transform=im_tfs)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)

# imgs = deprocess_img(train_data.__iter__().__next__()[0].view(128, 784)).numpy().squeeze()  # 可视化图片效果
# print(imgs.shape)
# show_images(imgs)


# 定义编码器
class Encoder(nn.Module):
    '''
    输入图像数据，得到方差值
    '''
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mean
        self.fc22 = nn.Linear(400, 20)  # var

    def encode(self, x):  # 编码层
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps.to(device))
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return z, KLD  # 返回编码数据z，与KLDloss


NOISE_DIM = 20


class Decoder(nn.Module):
    '''
    输入数据，得到图像
    '''
    def __init__(self, noise_dim=NOISE_DIM):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),  # 128,64,3,3
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),  # 128,1,1,1
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7)  # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    '''
    判别器，输入图像判
    '''
    def __init__(self):

        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),  # 24,24
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),  # 12,12
            nn.Conv2d(32, 64, 5, 1),  # 8,8
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)  # 4,4

        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.f = nn.Sequential(
            nn.Conv2d(64, 1, 4, 2, 1),
            nn.AvgPool2d(2)
        )  # 1,1

    def forward(self, x):
        x = self.conv(x)
        f = self.f(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, f.squeeze()  # f.squeeze() # 把f中所有维度为“1”的压缩


vae_Enc = Encoder().to(device)
vae_Dec = Decoder().to(device)
D = Discriminator().to(device)
E_trainer = torch.optim.Adam(vae_Enc.parameters(), lr=0.001)  # lr=0.001
G_trainer = torch.optim.Adam(vae_Dec.parameters(), lr=0.0005)  # lr=0.0003  betas=(0.5, 0.999)
D_trainer = torch.optim.Adam(D.parameters(), lr=0.0005)  # lr=0.0003


def lossD(label_origin, label_noise, label_vae_Dec):
    # 各项数据平方和*0.5再取均值
    loss = 0.5 * ((label_origin - 1) ** 2).mean() + 0.5 * (label_noise ** 2).mean() + 0.5 * (label_vae_Dec ** 2).mean()
    return loss


def lossGD(label_noise):
    loss = 0.5 * ((label_noise - 1) ** 2).mean()
    return loss


def training(epochs=10):

    for epoch in range(epochs):

        # i = 0
        for img_origin, _ in train_data:
            # i += 1
            # if i == 469:
            #     break
            img_origin = img_origin.to(device)
            bs = img_origin.shape[0]  # 获取当前的数据集量128
            x_r0 = img_origin.view(bs, -1)  # 数据格式转换
            z, _ = vae_Enc(x_r0)  # 返回编码数据、kldloss
            img_vae_Dec = vae_Dec(z)  # 得到解码图像
            sample_noise = (torch.rand(bs, NOISE_DIM) - 0.5) / 0.5  # 生成(bs, NOISE_DIM)组随机数据集并归一化
            g_fake_seed = Variable(sample_noise).to(device)
            img_nose_Dec = vae_Dec(g_fake_seed)  # 通过解码网络，得到假图像
            label_origin_Linear, _ = D(img_origin)  # 源图像 放入判别器
            label_vae_Dec_Linear, _ = D(img_vae_Dec)  # 正常编码再解码的图像 放入判别器  label图片判定的true与false
            label_noise_Linear, _ = D(img_nose_Dec)  # 随机噪声解码图 放入判别器

            # ---------------------识别器训练 ---------------------------
            # 识别器的loss为识别三次的累计
            loss_D = lossD(label_origin_Linear, label_noise_Linear, label_vae_Dec_Linear)
            D_trainer.zero_grad()
            loss_D.backward()
            D_trainer.step()

            # ------------------------G & E  training------------------
            z1, kld1 = vae_Enc(x_r0)
            img_vae_Dec1 = vae_Dec(z1)
            sample_noise = (torch.rand(bs, NOISE_DIM) - 0.5) / 0.5
            g_fake_seed = Variable(sample_noise).to(device)
            img_nose_Dec1 = vae_Dec(g_fake_seed)
            _, label_con_origin = D(img_origin)
            _, label_con_vae_Dec = D(img_vae_Dec1)
            label_noise_Linear, _ = D(img_nose_Dec1)
            loss_GD = lossGD(label_noise_Linear)
            # data.pow(n)幂次运算,data中每个值都**n
            data1 = (img_vae_Dec1 - img_origin).pow(2).mean()
            data2 = (label_con_vae_Dec - label_con_origin).pow(2).mean()
            loss_G = 0.5 * (data1 + data2)
            G_trainer.zero_grad()
            E_trainer.zero_grad()
            kld1.backward(retain_graph=True)  # 编码器的loss=kld1
            # 如果你的Y值是个标量，那么直接使用:y.backward()就可以了.
            # 但是如果你的Y是个向量或矩阵，那么就不一样:y.backward(torch.ones_like(x))
            loss_all = 0.01 * loss_G + loss_GD
            loss_all.backward(torch.ones_like(loss_G))
            G_trainer.step()
            E_trainer.step()
        imgs_numpy = deprocess_img(img_nose_Dec1.data.cpu().numpy())
        print('epoch: {}, loss_D: {:.4}, loss_GD:{:.4}'.format(epoch, loss_D.item(), loss_GD.item()))
        show_images(imgs_numpy[0:16])
        plt.show()


training()


