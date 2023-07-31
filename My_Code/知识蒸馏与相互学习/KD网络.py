# -*- coding: utf-8 -*-
import torchvision
from torchvision.models.resnet import resnet18, resnet50
import torch
from torchvision.transforms import transforms
import torchvision.datasets as dst
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn

resnet18_pretrain_weight = r"E:\data_all\CIFAR\cifar-10\weight"
resnet50_pretrain_weight = r"E:\data_all\CIFAR\cifar-10\weight"
img_dir = r"E:\data_all\CIFAR"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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
    return train_loader, test_loader, len(train_data)


def load_checkpoint(net, pth_file, exclude_fc=False):
    """
    加载训练断点参数
    :param net:
    :param pth_file:
    :param exclude_fc:排除函数
    """
    if exclude_fc:
        model_dict = net.state_dict()
        pretrain_dict = torch.load(pth_file)
        new_dict = {k: v for k, v in pretrain_dict.items() if 'fc' not in k}
        model_dict.update(new_dict)
        net.load_state_dict(model_dict, strict=True)
    else:
        pretrain_dict = torch.load(pth_file)
        net.load_state_dict(pretrain_dict, strict=True)


def accuracy(output, target, topk=(1,)):
    """计算指定值k的精度@k，等于1时精度要求最高"""
    maxk = max(topk)
    batch_size = target.size(0)
    # output.topk()函数取指定维度上的最大值(或最大maxk个)，第二个参数dim = 1，为按行取
    _, pred = output.topk(maxk, 1, True, True)  # 返回数据、数据索引
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


def test(net, test_loader):
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
        img = img.to(device)
        target = target.to(device)

        with torch.no_grad():
            out = net(img)
        prec1, prec5 = accuracy(out, target, topk=(1, 5))  # 此处得到的prec是每个bitch_size的ACC平均值
        prec1_sum += prec1
        prec5_sum += prec5
        # bitch_size*(i + 1)为样本总数len(test_loader)
    print(f"val_Acc1:{prec1_sum / (i + 1)}, valAcc5: {prec5_sum / (i + 1)}")


def train(net_s, net_t, train_loader, test_loader):
    """
    训练
    :param net_s: 学生网络模型
    :param net_t: 教师网络模型
    :param train_loader: 训练数据
    :param test_loader: 测试数据
    """
    # opt = Adam(filter(lambda p: p.requires_grad,net.parameters()), lr=0.0001)
    opt = Adam(net_s.parameters(), lr=0.0001)
    net_s.train()
    net_t.eval()
    for epoch in range(100):
        loss_all = 0
        prec_all = 0
        for step, batch in enumerate(train_loader):
            opt.zero_grad()
            image, target = batch  # 获取图像数据与对应的标签
            image = image.to(device)  # 数据传入gpu
            target = target.to(device)  # 数据传入gpu
            # forward pass
            out_s, out_t = net_s(image), net_t(image)  # 我们通过模型的每一层运行输入数据以进行预测。 这是正向传播
            loss_init = CrossEntropyLoss()(out_s, target)  # 求学生网络与标签的交叉熵损失
            loss_kd = KD_loss(T=2)(out_s, out_t)
            loss = loss_init + loss_kd
            loss_all += loss  # 不进行累加就只是显示当前step的loss而不是均值LOSS
            prec1, prec5 = accuracy(out_s, target, topk=(1, 5))
            out = out_s.argmax(1)
            acc = (out == target).sum()
            prec_all += prec1
            # 我们使用模型的预测和相应的标签来计算误差（loss）。
            # 下一步是通过网络反向传播此误差。 当我们在误差张量上调用.backward()时，开始反向传播。
            # 然后，Autograd会为每个模型参数计算梯度并将其存储在参数的.grad属性中。
            opt.zero_grad()  # 当前模型的梯度清零
            loss.backward()
            # 我们调用.step()启动梯度下降。 优化器通过.grad中存储的梯度来调整每个参数
            opt.step()
        # print(f"loss_all:{loss_all/(step+1)}")
        print(f"\nepoch:{epoch},train_loss: {loss_all / (step + 1)}, train_acc:{prec_all / (step + 1)}")
        test(net_s, test_loader)

    torch.save(net_s.state_dict(), './net18_cifar10_kd.pth')


def main():
    net_s = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net_s.fc = nn.Linear(net_s.fc.in_features, 10)  # 想输出为10个类别时
    net_t = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net_t.fc = nn.Linear(net_t.fc.in_features, 10)  # 想输出为10个类别时
    net_t.to(device)
    net_s.to(device)
    # load_checkpoint(net_t, resnet50_pretrain_weight, exclude_fc=False)
    # load_checkpoint(net_s, resnet18_pretrain_weight, exclude_fc=True)
    # 固定部分层参数(此处表示除fc层，其他层参数全部固定)
    for name, params in net_s.named_parameters():
        if 'fc' not in name:
            params.requires_grad = False  # 固定参数

    train_loader, test_loader = create_data(img_dir)

    train(net_s, net_t, train_loader, test_loader)


if __name__ == "__main__":

    main()
    # output = torch.rand(4, 10)
    # label = torch.Tensor([2, 1, 8, 5]).unsqueeze(dim=1)
    # print(output)
    # print('*' * 100)
    # values, indices = torch.topk(output, k=2, dim=1, largest=True, sorted=True)
    # print("values: ", values)
    # print("indices: ", indices)
    # print('*' * 100)
    #
    # print(accuracy(output, label, topk=(1, 2)))

