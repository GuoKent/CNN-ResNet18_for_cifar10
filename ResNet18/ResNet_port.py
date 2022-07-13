from dataloader import my_dataset
from ResNet import ResNet18
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 申明GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 导入ResNet模型
ResNet = ResNet18().to(device)
ResNet.load_state_dict(torch.load('ResNet18 epoch=135 lr=0.001 batch_size=32.pkl'))

# 准备数据集并预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

store_path = './cifar'
label = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
         'truck': 9}
batch_size = 16

# 定义损失函数
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题

# 测试集
split = 'test'
test_dataset = my_dataset(store_path, split, label, transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 开始测试
ResNet_error = 0
ResNet_loss = 0

for i, (inputs, labels) in enumerate(test_loader):
    with torch.no_grad():  # 测试网络时不需要梯度，使用这句防止内存被占用导致无法运行
        inputs, labels = Variable(inputs), Variable(labels)  # 转化为变量，里面的值会随时改变，用于迭代
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据放入GPU运算

        # 记录ResNet错误个数
        ResNet_outputs = ResNet(inputs)
        ResNet_outputs_list = ResNet_outputs.data.tolist()
        ResNet_outputs_labels = []
        ResNet_loss += criterion(ResNet_outputs, labels)  # 计算损失值
        for j in range(batch_size):  # 逐行找分类结果
            if j >= len(ResNet_outputs_list):
                break  # 数据集不能杯batch_size整除
            result = ResNet_outputs_list[j].index(max(ResNet_outputs_list[j]))
            ResNet_outputs_labels.append(result)  # 找出数值最大所在的标签
            if ResNet_outputs_labels[j] != labels.data.tolist()[j]:
                ResNet_error += 1  # 记录错误个数

total = len(test_dataset)  # 测试样本总数
ResNet_rate = (total - ResNet_error) / total  # ResNet准确率
ResNet_error_rate = ResNet_error / total  # ResNet错误率

print('测试样本总数：%d' % total)
print('ResNet分类正确个数：%d' % (total - ResNet_error))
print('ResNet分类错误个数：%d' % ResNet_error)
print('ResNet分类准确率：%f' % ResNet_rate)
print('ResNet分类错误率：%f' % ResNet_error_rate)
print('损失函数值loss：%f' % (ResNet_loss / total))
