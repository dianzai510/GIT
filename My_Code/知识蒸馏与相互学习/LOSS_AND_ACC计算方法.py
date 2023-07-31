# -*- coding: utf-8 -*-

# for epoch in range(num_epochs):
#     net.train()
#     train_total = 0
#     train_correct = 0
#     train_loss = 0
#     for batch in train_loader:
#         imgs, labels = batch
#         imgs, labels = imgs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         optputs = net(imgs)
#         loss = loss_function(optputs, labels)#获取单个loss
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()  # 获取总loss
#
#         predicted = torch.argmax(optputs, 1)  # torch.argmax()函数：获取当前索引最大数值
#         train_correct += (predicted == labels).sum().item()
#         train_total += labels.size(0)

'''
# 输出的predicted = torch.max(output.data, 1)[1]，解释代码：这里第一个1表示求行的最大值因为每一行代表了一个样本的输出结果，第二个1代表了我们只要索引，
        # 还有另一种方法：predicted = torch.argmax(output, 1)，torch.argmax()函数：求最大数值的索引
        # 接下来我们需要计算预测正确的个数：correct += (output == labels).sum().item()，首先，“outpuut == labels” 的语法求出正确的类，类似于[1, 0, 1, 0]
        # 1表示预测正确，0表示错误，然后.sum()将所有正确的预测加起来，得到预测正确的个数，
        # torch.item()，这时候输出的是一个tensor类，比如有两个预测正确：tensor(2)，
        # .item()语法将tensor转化为普通的float或者int

'''
# scheduler.step()
# train_loss = train_loss / len(train_loader)
# train_accuracy = train_correct / train_total
# print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_accuracy:.5f}")
