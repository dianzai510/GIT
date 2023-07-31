# -*- coding: utf-8 -*-
# 【torch.nn.functional.kl_div()】为【torch.nn.KLDivLoss()】的简化版，两个函数用法作用相同
# reduction：指定损失输出的形式，有四种选择：none|mean|batchmean|sum。
# none：损失不做任何处理，直接输出一个数组；
# mean：将得到的损失求平均值再输出，会输出一个数；
# batchmean：将输出的总和除以batchsize；
# sum：将得到的损失求和再输出，会输出一个数
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

# x = torch.randn((1, 8))
# print(x)
x = torch.Tensor([[1, 1, 2, 2, 2, 2, 3]])
# y = torch.randn((1, 8))
# print(y)
y = torch.Tensor([[1, 1, 2, 3, 3, 3, 3]])
# 先转化为概率，之后取对数
x_log = F.log_softmax(x/2, dim=1)
# 只转化为概率
y = F.softmax(y/2, dim=1)
kl = nn.KLDivLoss(reduction='batchmean')
out = kl(x_log, y)*2*2
print(x_log)
print(y)
print(out)

# # 如果model的最后一层有softmax，我们就相当于拿到了概率Q
# predicted = model(samples)  # 模型得到各类的概率
# # 取对数
# log_pre = torch.log(predicted)  # 对概率取对数值概
# # 只转化为概率
# labels = F.softmax(labels)
# loss = F.kl_div(log_pre, labels, reduction='batchmean')
#
#
# ####################################################
# # 如果model的最后没有softmax，代码如下
# predicted = model(samples)
# # 先转化为概率，之后取对数
# log_pre = nn.LogSoftmax(dim=1)(predicted)
# # 只转化为概率
# labels = F.softmax(labels)
# loss = F.kl_div(log_pre, labels, reduction='batchmean')
#

