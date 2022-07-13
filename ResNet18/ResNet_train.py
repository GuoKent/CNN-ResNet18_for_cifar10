import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ResNet import ResNet18
from dataloader import my_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 申明GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
])

# 测试集和训练集分开预处理
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

store_path = './cifar'  # 这里修改路径
label = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
         'truck': 9}
batch_size = 64  # 这里修改batch_size
epoch = 20  # 这里修改epoch
lr = 0.01  # 这里修改学习率

# 训练集
split = 'train'
train_dataset = my_dataset(store_path, split, label, transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# 测试集
split = 'test'
test_dataset = my_dataset(store_path, split, label, transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络，并移到GPU训练
ResNet = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题
# 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(ResNet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# 开始训练
train_error = []  # 训练误差，画图用
test_error = []  # 测试误差，画图用
num = []  # epoch次数
for epoch in range(epoch):
    num.append(epoch + 1)
    # 训练集训练网络
    loss = 0.0
    error = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = Variable(inputs), Variable(labels)  # 转化为变量，里面的值会随时改变，用于迭代
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据放入GPU运算

        optimizer.zero_grad()  # 梯度清零
        outputs = ResNet(inputs)  # 得到输出
        loss = criterion(outputs, labels)  # 计算损失值
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        outputs_list = outputs.data.tolist()  # 取出outputs里面的值并转化为list
        out_label = []  # 储存输出结果
        # print(len(outputs_list))

        for j in range(batch_size):  # 逐行找分类结果,range里面的数值是batch_size
            if j >= len(outputs_list):
                break  # 数据集不能杯batch_size整除
            result = outputs_list[j].index(max(outputs_list[j]))
            out_label.append(result)  # 找出数值最大所在的标签
            if out_label[j] != labels.data.tolist()[j]:
                error += 1  # 记录错误个数
    total = len(train_dataset)
    error_rate = error / total
    print('epoch=', epoch + 1)
    print('train_error_rate=', error_rate)
    train_error.append(error_rate)  # 将错误率添加到列表中

    # 测试集测试网络
    error = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = Variable(inputs), Variable(labels)  # 转化为变量，里面的值会随时改变，用于迭代
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据放入GPU运算

        outputs = ResNet(inputs)
        outputs_list = outputs.data.tolist()  # 取出outputs里面的值并转化为list
        out_label = []  # 储存输出结果
        for j in range(batch_size):  # 逐行找分类结果
            if j >= len(outputs_list):
                break  # 数据集不能杯batch_size整除
            result = outputs_list[j].index(max(outputs_list[j]))
            out_label.append(result)  # 找出数值最大所在的标签
            if out_label[j] != labels.data.tolist()[j]:
                error += 1  # 记录错误个数
    total = len(test_dataset)
    error_rate = error / total
    print('test_error_rate=', error_rate)
    test_error.append(error_rate)  # 将错误率添加到列表中

# 保存模型
torch.save(ResNet.state_dict(), 'ResNet18 2fc.pkl')
# 画图
plt.plot(num, train_error, c='red', label='train_error')
plt.plot(num, test_error, c='blue', label='test_error')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error_rate')
plt.show()
