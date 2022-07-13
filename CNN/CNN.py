import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import my_dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    # transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
])

store_path = './cifar'  # 这里修改路径
label = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
         'truck': 9}

# 训练集
split = 'train'
train_dataset = my_dataset(store_path, split, label, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

# 测试集
split = 'test'
test_dataset = my_dataset(store_path, split, label, transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# 定义网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 15, 3) # 卷积层1,第一个参数时输入channel,第二个是输出channel,第三个是卷积核大小
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(15, 75, 4)  # 卷积层2
        self.conv3 = nn.Conv2d(75, 375, 3)
        '''self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)'''
        # 结构2
        self.fc1 = nn.Linear(1500, 400)  # 接着三个全连接层
        self.fc2 = nn.Linear(400, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3通道变为6通道
        x = self.pool(F.relu(self.conv2(x)))  # 6通道变为16通道
        x = self.pool(F.relu(self.conv3(x)))  # 新增
        # x = x.view(-1, 16 * 5 * 5)  # tensor方法，将矩阵拉伸成一维向量
        x = x.view(-1, 1500)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 新增
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    # 申明GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    CNN = CNN()
    CNN.to(device)  # 将模型放入GPU运算
    criterion = nn.CrossEntropyLoss().to(device)  # 定义损失函数，Softmax–Log–NLLLoss三合一的损失函数，用于多分类
    optimizer = optim.SGD(CNN.parameters(), lr=0.0001, momentum=0.9)  # 快速梯度下降法

    # 开始迭代
    train_error = []  # 训练误差，画图用
    test_error = []  # 测试误差，画图用
    num = []  # epoch次数
    for epoch in range(50):
        num.append(epoch + 1)
        # 训练集训练网络
        loss = 0.0
        error = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)  # 转化为变量，里面的值会随时改变，用于迭代
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据放入GPU运算

            optimizer.zero_grad()  # 将梯度清零，否则下次循环会累加
            outputs = CNN(inputs)  # 输出是一个4*10的tensor，每一行是softmax的计算结果，每一列代表分类的种类
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 迭代器更新
            # print(loss)

            outputs_list = outputs.data.tolist()  # 取出outputs里面的值并转化为list
            out_label = []  # 储存输出结果
            for j in range(4):  # 逐行找分类结果
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

            outputs = CNN(inputs)
            outputs_list = outputs.data.tolist()  # 取出outputs里面的值并转化为list
            out_label = []  # 储存输出结果
            for j in range(4):  # 逐行找分类结果
                result = outputs_list[j].index(max(outputs_list[j]))
                out_label.append(result)  # 找出数值最大所在的标签
                if out_label[j] != labels.data.tolist()[j]:
                    error += 1  # 记录错误个数
        total = len(test_dataset)
        error_rate = error / total
        print('test_error_rate=', error_rate)
        test_error.append(error_rate)  # 将错误率添加到列表中

    # 保存模型
    torch.save(CNN.state_dict(), 'CNN no transform 3fc.pkl')
    # 画图
    plt.plot(num, train_error, c='red', label='train_error')
    plt.plot(num, test_error, c='blue', label='test_error')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('error_rate')
    plt.show()
