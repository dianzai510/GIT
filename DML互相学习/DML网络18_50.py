# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import shutil

import torchvision
from torchvision.models.resnet import resnet18, resnet50
import torch
from torchvision.transforms import transforms
import torchvision.datasets as dst
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn.functional as F
import torch.nn as nn

resnet18_pretrain_weight = r"E:\data_all\CIFAR\cifar-10\weight"
resnet50_pretrain_weight = r"E:\data_all\CIFAR\cifar-10\weight"
img_dir = r"E:\data_all\CIFAR"


def val(net, test_loader):
    '''
    验证
    :param net:
    :param test_loader:
    :return:
    '''
    prec1_sum = 0
    prec5_sum = 0
    net.eval()
    for i, (img, target) in enumerate(test_loader, start=1):
        # print(f"batch: {i}")
        img = img.cuda()
        target = target.cuda()
        with torch.no_grad():
            out = net(img)
        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        prec1_sum += prec1
        prec5_sum += prec5
        # print(f"batch: {i}, acc1:{prec1}, acc5:{prec5}")
    print(f"Acc1:{prec1_sum / (i + 1)}, Acc5: {prec5_sum / (i + 1)}")


class KD_loss(nn.Module):
    def __init__(self, T):
        super(KD_loss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        # 因为要用t指导s,所以求s的对数概率，t的概率
        logp_s = F.log_softmax(out_s / self.T, dim=1)
        p_t = F.softmax(out_t / self.T, dim=1)
        loss = F.kl_div(logp_s, p_t, reduction='batchmean') * self.T * self.T

        return loss


def create_data(img_dir):
    '''
    :param img_dir:
    :return: 返回训练数据与测试数据集
    '''
    dataset = dst.CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    data_transform = {  # 数据预处理
        "train": transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        "val": transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])}
    # 加载数据集，指定训练或测试数据，指定于处理方式
    train_data = dataset(root=img_dir, train=True, transform=data_transform["train"], download=True)
    test_data = dataset(root=img_dir, train=False, transform=data_transform["val"], download=True)

    # train_transform = transforms.Compose([
    #     transforms.Pad(4, padding_mode='reflect'),
    #     transforms.RandomCrop(32),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])
    # test_transform = transforms.Compose([
    #     transforms.CenterCrop(32),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    # define data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def accuracy(output, target, topk=(1, 2)):
    """
    计算指定值k的精度
    需要计算top_k准确率中的k值，元组类型。默认为(1, 5)，即函数返回top1和top5的分类准确率
    top1比top5要求更高，top5表示取五个最大值，五个值中只要有一个与标签位置对上就过
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # output.topk()函数取指定维度上的最大值(或最大maxk个)，第二个参数dim = 1，为按行取
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置
    target2 = target.view(1, -1)  # view函数作用是将张量铺平
    # 使用target2.expand_as(pred)将张量arget2的size——torch.Size([3, 1])扩展为与张量pred的size——torch.Size([3, 2])同形的高维张量
    #  torch.eq()函数就是用来比较对应位置数字，相同则为1，否则为0，输出与那两个tensor大小相同，并且其中只有1和0
    correct = pred.eq(target2.expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k]
        #  view()需要Tensor中的元素地址是连续的，Tensor不连续的情况会报错，加入.contiguous()可避免报错。
        correct_s = correct_k.contiguous().view(-1)
        correct_s = correct_s.float().sum(0, keepdim=True)
        res.append(correct_s.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    计算并存储平均值和当前值。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(i, state, is_best):
    """
    保存模型副本，以便将来加载。该函数用于在测试数据上评估模型。
    如果该模型达到了迄今为止的最佳验证精度，则会创建一个后缀为 `best` 的单独文件。
    """
    # print("[*] Saving model to {}".format(self.ckpt_dir))
    model_name = "save_model"
    ckpt_dir = "model_dir"
    filename = model_name + str(i + 1) + '_ckpt.pth'
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

    if is_best:
        filename = model_name + str(i + 1) + '_model_best.pth'
        shutil.copyfile(
            ckpt_path, os.path.join(ckpt_dir, filename)
        )


def load_checkpoint(best=False):
    """
        加载模型的最佳副本。这在以下两种情况下非常有用
        - 加载最佳验证模型，以便在测试数据上进行评估。
        参数 ------ - best：如果设置为 True，则加载最佳模型。
        如果要在测试数据上评估模型，请使用此参数。否则，设置为 "假"，则使用最新版本的检查点。
        """
    model_name = "save_model"
    ckpt_dir = "model_dir"
    filename = model_name + '_ckpt.pth'
    if best:
        filename = model_name + '_model_best.pth'
    ckpt_path = os.path.join(ckpt_dir, filename)
    ckpt = torch.load(ckpt_path)
    # 从检查点加载变量
    epoch = ckpt['epoch']
    best_valid_acc = ckpt['best_valid_acc']
    # model.load_state_dict(ckpt['model_state'])
    # optimizer.load_state_dict(ckpt['optim_state'])
    model = (ckpt['model_state'])
    optimizer = (ckpt['optim_state'])
    if best:
        print("[*] Loaded {} checkpoint @ epoch {} ""with best valid acc of {:.3f}".format
              (filename, ckpt['epoch'], ckpt['best_valid_acc']))
    else:
        print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt['epoch']))


def loss_and_acc(net1, net2, label):
    '''
    获取损失结果与精度
    :param net1:学习网络
    :param net2:被学习网络
    :param label: 标签
    :return:
    '''
    ce_loss = nn.CrossEntropyLoss()(net1, label)  # 当前模型的交叉熵损失值
    kl_loss = KD_loss(T=2)(net1, net2)
    loss = ce_loss + kl_loss  # 当前模型最后的loss
    # 测量精度并记录应力损失:
    prec = accuracy(net1.data, label.data, topk=(1, 5))[0]  # 此处得到的prec是每个bitch_size的ACC平均值
    return loss, prec


def train_dml(train_loader, test_loader):
    '''
    相互学习网络(通过循环编写程序)
    :param net_all: 需要相互学习的网络的列表
    :param train_loader: 数据集
    :param test_loader: 测试集合
    :return:
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models_all = []
    best_valid_accs = [0.] * len(models_all)
    # 定义模型列表
    # model = models.vgg16_bn(pretrained=True).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    net_s = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net_s.fc = nn.Linear(net_s.fc.in_features, 10)  # 想输出为10个类别时
    net_t = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net_t.fc = nn.Linear(net_t.fc.in_features, 10)  # 想输出为10个类别时
    # .cuda()和.to(device)的效果一样吗？为什么后者更好？
    # 两个方法都可以达到同样的效果，在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。
    # 调用model.cuda()，可以将模型加载到GPU上去。这种方法不被提倡，而建议使用model.to(device)的方式，
    # 这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下。
    net_t = net_t.to(device)
    net_s = net_s.to(device)
    models_all.append(net_s)  # 加入第1个模型
    models_all.append(net_t)  # 加入第2个模型,后续可继续添加
    best_valid_accs = [0.] * len(models_all)
    # 确定优化器和优化策略
    opt_all = []
    schedulers_all = []
    for i in range(len(models_all)):
        # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
        opt = SGD(models_all[i].parameters(), lr=0.001)
        opt_all.append(opt)
        # 设置学习衰减率   set learning rate decay
        scheduler = lr_scheduler.StepLR(opt_all[i], step_size=60, gamma=1, last_epoch=-1)
        schedulers_all.append(scheduler)
    for epoch in range(100):  # 循环训练epoch遍
        print('\nEpoch: {} - LR: {:.4f}'.format(epoch, opt_all[0].param_groups[0]['lr']))
        ####### 训练模型 #########
        train_losses = []
        train_accs = []
        for i in range(len(models_all)):
            models_all[i].train()
            train_losses.append(AverageMeter())
            train_accs.append(AverageMeter())
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = []
            for model in models_all:  # 一个模型
                outputs.append(model(images))  # 图片传入网络得到out值
            # ============================ 损失函数 ============================
            # 这个循环结束就是所有模型训练了一遍，即整个互学习模型训练了一遍，即以前那种单个模型训练一遍
            for i in range(len(models_all)):
                opt_all[i].zero_grad()  # 当前模型的梯度清零
                ce_loss = nn.CrossEntropyLoss()(outputs[i], labels)  # 当前模型的交叉熵损失值
                kl_loss = 0
                for j in range(len(models_all)):  # KL散度 重点
                    if i != j:
                        kl_loss += F.kl_div(F.log_softmax(outputs[i] / 2, dim=1),  # lenet
                                            F.softmax(outputs[j] / 2, dim=1), reduction='batchmean') * 2 * 2
                loss = ce_loss + kl_loss / (len(models_all) - 1)  # 当前模型最后的loss
                # 测量精度并记录应力损失:
                prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]  # 此处得到的prec是每个bitch_size的ACC平均值
                train_losses[i].update(loss.item(), images.size()[0])  # 记录每个模型的loss
                train_accs[i].update(prec.item(), images.size()[0])  # 记录每个模型的acc
                # 计算梯度并更新 SGD
                # ============================ 优化器 ==============================
                loss.backward(retain_graph=True)  # loss反向传播
                opt_all[i].step()  # 当前模型的优化器进行优化
                schedulers_all[i].step()  # 模型的优化

        ####### 验证模型 #######
        valid_losses = []
        valid_accs = []
        for i in range(len(models_all)):
            models_all[i].eval()  # 模型参数固化
            valid_losses.append(AverageMeter())
            valid_accs.append(AverageMeter())
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = []
            for model in models_all:
                outputs.append(model(images))
            # ============================ 损失函数 ============================
            for i in range(len(models_all)):
                ce_loss2 = nn.CrossEntropyLoss()(outputs[i], labels)
                kl_loss2 = 0
                for j in range(len(models_all)):
                    if i != j:  # 其他模型的的
                        kl_loss2 += F.kl_div(F.log_softmax(outputs[i] / 2, dim=1),
                                             F.softmax(outputs[j] / 2, dim=1), reduction='batchmean') * 2 * 2
                loss2 = ce_loss2 + kl_loss2 / (len(models_all) - 1)
                # 测量精度和记录损失
                prec = accuracy(outputs[i].data, labels.data, topk=(1,))[0]  # 此处得到的prec是每个bitch_size的ACC平均值
                valid_losses[i].update(loss2.item(), images.size()[0])  # update 类似于 append
                valid_accs[i].update(prec.item(), images.size()[0])
        #  数据保存显示
        for i in range(len(models_all)):
            is_best = valid_accs[i].avg > best_valid_accs[i]
            msg1 = "model_{:d}: train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                # self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(i + 1, train_losses[i].avg, train_accs[i].avg, valid_losses[i].avg, valid_accs[i].avg))
            # best_valid_accs[i] = max(valid_accs[i].avg, best_valid_accs[i])
            # save_checkpoint(i, {'epoch': epoch + 1,
            #                     'model_state': models_all[i].state_dict(),
            #                     'optim_state': opt_all[i].state_dict(),
            #                     'best_valid_acc': best_valid_accs[i], }, is_best)


def train_dml_2(train_loader, test_loader):
    '''
    相互学习网络——直接手写
    :param train_loader: 数据集
    :param test_loader: 测试集合
    :return:
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_s = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net_s.fc = nn.Linear(net_s.fc.in_features, 10)  # 想输出为10个类别时
    net_t = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net_t.fc = nn.Linear(net_t.fc.in_features, 10)  # 想输出为10个类别时
    net_t.to(device)
    net_s.to(device)
    # 确定优化器和优化策略
    opt_t = SGD(net_t.parameters(), lr=0.001)
    opt_s = SGD(net_s.parameters(), lr=0.001)
    scheduler_t = lr_scheduler.StepLR(opt_t, step_size=60, gamma=1, last_epoch=-1)
    scheduler_s = lr_scheduler.StepLR(opt_s, step_size=60, gamma=1, last_epoch=-1)
    for epoch in range(100):  # 循环训练epoch遍
        print('\nEpoch: {} - LR: {:.4f}'.format(epoch, opt_s.param_groups[0]['lr']))
        # 训练模型
        net_t.train()
        net_s.train()
        train_loss_s = 0
        train_acc_s = 0
        train_loss_t = 0
        train_acc_t = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch = labels.size(0)
            opt_s.zero_grad()  # 当前模型的梯度清零
            opt_t.zero_grad()  # 当前模型的梯度清零
            out_s, out_t = net_s(images), net_t(images)
            # ============================ 损失函数 ============================
            # 学生网络学习老师网络
            loss_s, prec_s = loss_and_acc(out_s, out_t, labels)  # 此处得到的prec是每个bitch_size的ACC平均值
            train_loss_s += loss_s / batch
            train_acc_s += prec_s
            # 计算梯度并更新 SGD
            # ============================ 优化器 ==============================
            loss_s.backward(retain_graph=True)  # loss反向传播
            opt_s.step()  # 当前模型的优化器进行优化
            scheduler_s.step()  # 模型的优化
            # 老师网络学习学生网络
            loss_t, prec_t = loss_and_acc(out_t, out_s, labels)
            train_loss_t += loss_t / batch
            train_acc_t += prec_t
            # 计算梯度并更新 SGD
            # ============================ 优化器 ==============================
            loss_t.backward()  # loss反向传播
            opt_t.step()  # 当前模型的优化器进行优化
            scheduler_t.step()  # 模型的优化
        num = len(train_loader)
        print("epoch:{}, train_loss_s:{:.4f}, train_acc_s: {:.4f}, train_loss_t:{:.4f}, train_acc_t:{:.4f}".format
              (epoch, train_loss_s.item() / num, train_acc_s.item() / num, train_loss_t.item() / num, train_acc_t.item() / num))
        ####### 验证模型 #######
        net_t.eval()  # 模型参数固化
        net_s.eval()  # 模型参数固化
        val_loss_s = 0
        val_acc_s = 0
        val_loss_t = 0
        val_acc_t = 0
        for k, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            batch = labels.size(0)  # 得到batch_size大小
            with torch.no_grad():
                out_s, out_t = net_s(images), net_t(images)

            loss_s_v, prec1 = loss_and_acc(out_s, out_t, labels)
            val_loss_s += loss_s_v / batch  # 得到每个loss均值，并累加
            val_acc_s += prec1

            loss_t_v, prec2 = loss_and_acc(out_t, out_s, labels)
            val_loss_t += loss_t_v / batch
            val_acc_t += prec2
        num = len(test_loader)
        print("epoch:{}, val_loss_s:{:.4f}, val_acc_s: {:.4f}, val_loss_t:{:.4f}, val_acc_t:{:.4f}".format
              (epoch, val_loss_s.item() / num, val_acc_s.item() / num, val_loss_t.item() / num, val_acc_t.item() / num))

    # 保存模型
    # torch.save(net_s.state_dict(), './resnet18_cifar10_kd.pth')
    torch.save({'model_state_dict': net_s.state_dict(),
                'epoch': epoch}, './resnet18_cifar10_kd.pth')


def main():
    train_loader, test_loader = create_data(img_dir)
    train_dml(train_loader, test_loader)
    # train_dml_2(train_loader, test_loader)


if __name__ == "__main__":
    main()
