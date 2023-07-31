#!/usr/bin/env python
# coding: utf-8
# 下面我们用 mnist 数据集来简单说明一下变分自动编码器
from torch.utils.data import DataLoader, sampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

img_dir = r"E:\data_all\MNIST"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置画图的尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


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


def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5


def deprocess_img(x):
    # return (x + 1.0) / 2.0
    return (x * 0.5) + 0.5


class ChunkSampler(sampler.Sampler):  # 定义一个取样的函数
    """从某个偏移量开始依次对元素进行采样.
    Arguments:
        num_samples: # 所需的数据点
        start: 我们应该从哪里开始选择
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


NUM_TRAIN = 50000
NUM_VAL = 5000
train_set = MNIST(img_dir, train=True, transform=preprocess_img)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)
# val_set = MNIST(img_dir, train=True, transform=preprocess_img)
# val_data = DataLoader(val_set, batch_size=128, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))


class VAE_con(nn.Module):

    def __init__(self, LATENT_CODE_NUM=20):
        super(VAE_con, self).__init__()
        # 如果编码层使用卷积层(如nn.Conv2d )，
        # 解码器需要使用反卷积层(nn.ConvTranspose2d)。
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True), )

        self.fc11 = nn.Sequential(nn.Linear(256 * 7 * 7, 256 * 7),
                                  nn.ReLU(),
                                  nn.Linear(256 * 7, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, LATENT_CODE_NUM), )
        self.fc12 = nn.Sequential(nn.Linear(256 * 7 * 7, 256 * 7),
                                  nn.ReLU(),
                                  nn.Linear(256 * 7, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, LATENT_CODE_NUM), )
        self.fc2 = nn.Sequential(nn.Linear(LATENT_CODE_NUM, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256 * 7),
                                 nn.ReLU(),
                                 nn.Linear(256 * 7, 256 * 7 * 7),
                                 nn.Tanh(), )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).to(device)
        z = mu + eps * torch.exp(logvar / 2)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        out11 = out1.view(out1.size(0), -1)  # 展开卷积后的数据
        out22 = out2.view(out2.size(0), -1)
        mu = self.fc11(out11)  # batch_s, latent
        logvar = self.fc12(out22)  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z)  # 数据经过全连接层
        out33 = out3.view(z.size(0), 256, 7, 7)  # 转换数据格式
        out4 = self.decoder(out33).view(out33.size(0), -1)  # 放入卷积层
        return out4, mu, logvar


net2 = VAE_con()  # 实例化网络
net2 = net2.to(device)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mean
        self.fc22 = nn.Linear(400, 20)  # var
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):  # 编码层
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # e**(x*0.5)
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps.to(device))
        return eps.mul(std).add_(mu)

    def decode(self, z):  # 解码层
        h3 = F.relu(self.fc3(z))
        return F.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        return self.decode(z), mu, logvar  # 解码，同时输出均值方差


net = VAE()  # 实例化网络
net = net.to(device)
reconstruction_function = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD


def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


for e in range(100):
    for im, _ in train_data:
        im1 = im.view(im.shape[0], -1)
        im1 = Variable(im1).to(device)
        im2 = Variable(im).to(device)
        # 不带卷积网络#（成像模糊）
        recon_im, mu, logvar = net(im1)
        loss = loss_function(recon_im, im1, mu, logvar) / im1.shape[0]  # 将 loss 平均
        # # 带卷积网络#（成像模糊）
        # recon_im, mu2, logvar2 = net2(im2)
        # loss = loss_function(recon_im, im2.view(im.shape[0], -1), mu2, logvar2) / im2.shape[0]  # 将 loss 平均

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch: {}, Loss: {:.4f}'.format(e, loss.item()))
    if e % 10 == 0:
        imgs_numpy = deprocess_img(recon_im.data.cpu().numpy())
        show_images(imgs_numpy[0:16])
        # imgs_numpy = deprocess_img(im1.data.cpu().numpy())
        # show_images(imgs_numpy[0:16])
        plt.show()
        print()
        # save = to_img(recon_im.cpu().data)
        # if not os.path.exists('./vae_img'):
        #     os.mkdir('./vae_img')
        # save_image(save, './vae_img/image_{}.png'.format(e + 1))
