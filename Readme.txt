dataloader.py 是数据集加载文件，要和其他文件放在同一个目录中
CNN.py 是训练CNN模型的文件，完整运行后会出现测试误差和训练误差的统计图
CNN_port.py 是CNN网络的测试接口，将训练好后的pkl模型文件与该文件放入同一个目录中，导入数据即可直接测试网络的效果
ResNet.py 是用于搭建ResNet18网络的文件，不需要运行，里面写了ResNet18的网络结构
ResNet_train.py 是ResNet网络的训练文件，完整运行后会出现测试误差和训练误差的统计图
ResNet_port.py 是ResNet网络的测试接口，将训练好后的pkl模型文件与该文件放入同一个目录中，导入数据即可直接测试网络的效果
压缩包的其他pkl文件时保存的网络模型文件，以网络结构和参数命名

GPU: RTX2070、RTX1660ti

依赖库：
torch=1.11.0+cu113
torchvision=0.12.0+cu113
matplotlib=2.2.3

在ResNet_port.py 第25行修改数据集路径即可运行
在CNN_port.py 第24行修改数据集路径即可运行