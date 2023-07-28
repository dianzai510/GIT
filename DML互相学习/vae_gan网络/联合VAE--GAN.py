#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image
# from google.colab import drive
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


def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5


def deprocess_img(x):
    return (x * 0.5) + 0.5


# 数据处理
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
train_set = MNIST(img_dir, train=True, download=True, transform=preprocess_img)
train_data = DataLoader(train_set, batch_size=128, sampler=ChunkSampler(NUM_TRAIN, 0))


# imgs = deprocess_img(train_data.__iter__().__next__()[0].view(batch_size, 784)).numpy().squeeze()  # 可视化图片效果
# show_images(imgs)


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

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True), )

        self.fc11 = nn.Linear(64 * 7 * 7, LATENT_CODE_NUM)
        self.fc12 = nn.Linear(64 * 7 * 7, LATENT_CODE_NUM)
        self.fc2 = nn.Linear(LATENT_CODE_NUM, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).to(device)
        z = mu + eps * torch.exp(logvar / 2)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 64, 7, 7)  # batch_s, 8, 7, 7
        out4 = self.decoder(out3) .view(out3.size(0), -1)
        return out4, mu, logvar


# VAE结构如下，也是生成器
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
        std = logvar.mul(0.5).exp_()  # e**(x/2)
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


# 判别器
def discriminator():
    net = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 1)
    )
    return net


class discriminator_con(nn.Module):
    def __init__(self):
        super(discriminator_con, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 784),
            nn.LeakyReLU(0.2),
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# 计算loss
def ls_discriminator_loss(scores_real, scores_fake):  # 判别器的loss
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss


def ls_generator_loss_vae(recon_x, x, mu, logvar):
    """
    recon_x: 生成图像
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    loss0 = 0.5 * ((recon_x - 1) ** 2).mean()
    # reconstruction_function = nn.MSELoss(reduction='sum')
    # loss0 = reconstruction_function(recon_x, x)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return loss0 + KLD


# 使用 adam 来进行训练，学习率是 3e-4, beta1 是 0.5, beta2 是 0.999
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer


def train_a_gan(D_net, G_vae_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss_vae):
    for epoch in range(100):
        for x, _ in train_data:
            bs = x.shape[0]  # 128
            # 判别网络
            real_data = Variable(x).view(bs, -1).to(device)  # 数据集由x：[128,1,28,28]变为real_data：[128,784]
            logits_real = D_net(real_data)  # 判别网络得分
            fake_images, mu, logvar = G_vae_net(real_data)  # 生成的假的数据
            logits_fake = D_net(fake_images)  # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake)  # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 生成网络
            fake_images, mu, logvar = G_vae_net(real_data)  # 生成的假的数据
            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss_vae(gen_logits_fake, real_data, mu, logvar)  # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()  # 优化生成网络
        print('epoch: {}, D: {:.4}, G:{:.4}'.format(epoch, d_total_error.item(), g_error.item()))
        if epoch % 5 == 0:
            imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
            show_images(imgs_numpy[0:16])
            plt.show()
            print()


def train_con_gan(D_net_con, G_vaecon_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss_vae):
    for epoch in range(30):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = Variable(x).to(device)  # 数据不进行展平
            logits_real = D_net_con(real_data)  # 判别网络得分
            fake_img, mu, logvar = G_vaecon_net(real_data)  # 生成的假的数据
            fake_data = fake_img.view((x.shape[0]), 1, 28, 28)
            logits_fake = D_net_con(fake_data)  # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake)  # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 生成网络
            fake_img, mu, logvar = G_vaecon_net(real_data)  # 生成的假的数据
            fake_data = fake_img.view((x.shape[0]), 1, 28, 28)
            gen_logits_fake = D_net_con(fake_data)
            g_error = generator_loss_vae(gen_logits_fake, real_data.view(bs, -1), mu, logvar)  # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()  # 优化生成网络
        print('epoch: {}, D: {:.4}, loss_vae:{:.4}'.format(epoch, d_total_error.item(), g_error.item()))
        if epoch % 5 == 0:
            imgs_numpy = deprocess_img(fake_img.data.cpu().numpy())
            show_images(imgs_numpy[0:16])
            plt.show()
            print()


# D_net = discriminator().to(device)
# G_vae_net = VAE().to(device)
# D_optim = get_optimizer(D_net)
# G_optim = get_optimizer(G_vae_net)
# # 普通vae网络训练（成像清晰）
# train_a_gan(D_net, G_vae_net, D_optim, G_optim, ls_discriminator_loss, ls_generator_loss_vae)


# 带卷积vae网络训练
D_net_con = discriminator_con().to(device)
G_vaecon_net = VAE_con().to(device)
D_optim_con = get_optimizer(D_net_con)
G_optim_con = get_optimizer(G_vaecon_net)
train_con_gan(D_net_con, G_vaecon_net, D_optim_con, G_optim_con, ls_discriminator_loss, ls_generator_loss_vae)
