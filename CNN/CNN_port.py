from dataloader import my_dataset
from CNN import CNN
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 申明GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 导入CNN模型
CNN = CNN().to(device)
CNN.load_state_dict(torch.load('CNN 80%.pkl'))

# 准备数据集并预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
])

store_path = '../cifar'  # 这里修改路径
label = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
         'truck': 9}
batch_size = 4

# 定义损失函数
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题

# 测试集
split = 'test'
test_dataset = my_dataset(store_path, split, label, transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 开始测试
CNN_error = 0
CNN_loss = 0

for i, (inputs, labels) in enumerate(test_loader):
    with torch.no_grad():  # 测试网络时不需要梯度，使用这句防止内存被占用导致无法运行
        inputs, labels = Variable(inputs), Variable(labels)  # 转化为变量，里面的值会随时改变，用于迭代
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据放入GPU运算

        # 记录CNN错误个数
        CNN_outputs = CNN(inputs)
        CNN_outputs_list = CNN_outputs.data.tolist()  # 取出outputs里面的值并转化为list
        CNN_outputs_labels = []
        CNN_loss += criterion(CNN_outputs, labels)  # 计算损失值
        for j in range(batch_size):  # 逐行找分类结果
            if j >= len(CNN_outputs_list):
                break  # 数据集不能杯batch_size整除
            result = CNN_outputs_list[j].index(max(CNN_outputs_list[j]))
            CNN_outputs_labels.append(result)  # 找出数值最大所在的标签
            if CNN_outputs_labels[j] != labels.data.tolist()[j]:
                CNN_error += 1  # 记录错误个数

total = len(test_dataset)  # 测试样本总数
CNN_rate = (total - CNN_error) / total  # CNN准确率
CNN_error_rate = CNN_error / total  # CNN错误率

print('测试样本总数：%d' % total)
print()
print('CNN分类正确个数：%d' % (total - CNN_error))
print('CNN分类错误个数：%d' % CNN_error)
print('CNN分类准确率：%f' % CNN_rate)
print('CNN分类错误率：%f' % CNN_error_rate)
print('损失函数值loss：%f' % (CNN_loss / total))
