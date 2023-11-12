# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 设定参数
PATH_DATASETS = "" # 预设路径
BATCH_SIZE = 1000  # 批量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"

image_width = 28
train_transforms = transforms.Compose([
    # 裁切部分图像，再调整图像尺寸
    transforms.RandomResizedCrop(image_width, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=(-10, 10)), # 旋转 10 度
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

test_transforms = transforms.Compose([
    transforms.Resize((image_width, image_width)), # 调整图像尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            batch = (batch_idx+1) * len(data)
            data_count = len(train_loader.dataset)
            percentage = (100. * (batch_idx+1) / len(train_loader))
            print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' +
                f'  Loss: {loss.item():.6f}')
    return loss_list

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if type(data) == tuple:
                data = torch.FloatTensor(data)
            if type(target) == tuple:
                target = torch.Tensor(target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    # 显示测试结果
    data_count = len(test_loader.dataset)
    percentage = 100. * correct / data_count
    print(f'准确率: {correct}/{data_count} ({percentage:.2f}%)')


if __name__ == '__main__':
    # 下载 MNIST 手写阿拉伯数字 训练资料
    train_ds = MNIST(PATH_DATASETS, train=True, download=True,
                    transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    # 下载测试资料
    test_ds = MNIST(PATH_DATASETS, train=False, download=True,
                    transform=test_transforms)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

    # 训练/测试资料的维度
    print(train_ds.data.shape, test_ds.data.shape)
    
    epochs = 10
    lr=1

    # 建立模型
    model = Net().to(device)

    # 损失
    criterion = F.nll_loss # nn.CrossEntropyLoss()

    # 设定优化器(optimizer)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

    loss_list = []
    for epoch in range(1, epochs + 1):
        loss_list += train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader)
        optimizer.step()

    # 模型存档
    torch.save(model, 'cnn_augmentation_model.pt')
       
    plt.plot(loss_list, 'r')
    plt.show()