import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F

# 设置随机种子
torch.manual_seed(42)
torch.cuda.set_device(0)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio, drop_connect_rate):
        super(MBConvBlock, self).__init__()
        
        self.expand_ratio = expand_ratio
        self.drop_connect_rate = drop_connect_rate
        
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(expanded_channels)
            self.act0 = Swish()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.act1 = Swish()
        
        # Squeeze and Excitation
        if se_ratio > 0:
            reduced_dim = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, reduced_dim, kernel_size=1),
                Swish(),
                nn.Conv2d(reduced_dim, expanded_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.has_skip = (in_channels == out_channels and stride == 1)
    
    def forward(self, x):
        identity = x
        
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.act0(x)
        
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        if hasattr(self, 'se'):
            x = x * self.se(x)
        
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        
        if self.has_skip and self.drop_connect_rate > 0:
            if self.training:
                keep_prob = 1 - self.drop_connect_rate
                random_tensor = keep_prob + torch.rand([x.size(0), 1, 1, 1], dtype=x.dtype, device=x.device)
                binary_tensor = torch.floor(random_tensor)
                x = x / keep_prob * binary_tensor
            else:
                x = x
        
        if self.has_skip:
            x = x + identity
        
        return x

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, kernel_size=3, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(16, 24, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(24, 24, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(24, 40, expand_ratio=6, kernel_size=5, stride=2, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(40, 40, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(40, 80, expand_ratio=6, kernel_size=3, stride=2, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(80, 80, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(80, 112, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(112, 112, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(112, 192, expand_ratio=6, kernel_size=5, stride=2, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(192, 192, expand_ratio=6, kernel_size=5, stride=1, se_ratio=0.25, drop_connect_rate=0.2),
            MBConvBlock(192, 320, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25, drop_connect_rate=0.2)
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class ImgDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # self.targets = targets
    
    def __getitem__(self, index):
        tmp_x = self.data[index]
        x = tmp_x['img']
        x = np.array(x, dtype=np.float32)
        new_order = (2, 0, 1)
        x = np.transpose(x, new_order)
        # 分离文件名和扩展名
        file_name, file_extension = os.path.splitext(tmp_x['filename'])
        
        y = file_name.split('_')
        y = float(y[-1])
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
    
    def __len__(self):
        return len(self.data)


def load_images_from_folder(folder):
    images = []
    for subject_name in os.listdir(folder):
        subject_dir_path = os.path.join(folder, subject_name)
        for filename in os.listdir(subject_dir_path):
            file_path = os.path.join(subject_dir_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                try:
                    with Image.open(file_path) as img:
                        # 处理图像
                        images.append({'img': img.copy(), 'filename': filename})
                except Exception as e:
                    print(f"Error opening {file_path}: {e}")
    return images

# 加载图片
image_path = '/gjh/ECGtoBP/ECG_images_with_bp'
img_obj_list = load_images_from_folder(image_path)

dataset = ImgDataset(img_obj_list)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 200
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数，MSE
loss_fn = nn.MSELoss()
# 实例化模型
model = EfficientNetB0(num_classes=1)
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# best_r2 = -float('inf')
best_mse = float('inf')
loss_list = []

from tqdm import tqdm
def train(model, device, train_loader, optimizer, criterion, epochs, eval_loader):
    # global best_r2
    global best_mse
    global loss_list
    print("training")
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        print('epoch:',epoch)
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清空梯度
            output = model(data)
            output = output.squeeze()
            loss = criterion(output, target)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            loss_list.append(loss.item())

        print('Loss: {:.6f}'.format( loss.item()))
        model.eval() # 评估模式
        print('评估模型在训练集上的性能：')
        with torch.no_grad():  # 不需要梯度计算
            predictions = []
            true_values = []
            for batch_idx, (inputs, labels) in enumerate(tqdm(eval_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())
                
        
            # 转换为NumPy数组以便计算
            predictions = np.array(predictions)
            true_values = np.array(true_values)

            # 计算评估指标
            mse = mean_squared_error(true_values, predictions)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(true_values - predictions))
            r2 = r2_score(true_values, predictions)

            if mse < best_mse:
                best_mse = mse
                # 保存模型的权重
                torch.save(model.state_dict(), '/gjh/ECGtoBP/best_EfficientNet_weights_41.pth')
            
            print(f'MSE: {mse:.4f}')
            print(f'RMSE: {rmse:.4f}')
            print(f'MAE: {mae:.4f}')
            print(f'R² Score: {r2:.4f}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 30
train(model, device, train_loader, optimizer, loss_fn, epochs, train_loader)

import matplotlib.pyplot as plt
# 绘制损失变化图
plt.figure(figsize=(15, 7))
plt.plot(loss_list, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Over Batches')
plt.legend()

# 保存图表为文件
plt.savefig('/gjh/ECGtoBP/EfficientNet_41_training_loss.png')

best_model = EfficientNetB0(num_classes=1)
best_model.load_state_dict(torch.load('/gjh/ECGtoBP/best_EfficientNet_weights_41.pth'))
best_model.to(device)

def test(model, device, test_loader):
    model.eval()
    print('评估模型在测试集上的性能：')
    with torch.no_grad():  # 不需要梯度计算
        predictions = []
        true_values = []
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(labels.cpu().numpy())            
    
        # 转换为NumPy数组以便计算
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        # 计算评估指标
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_values - predictions))
        r2 = r2_score(true_values, predictions)

        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'R² Score: {r2:.4f}')
    
test(best_model, device, test_loader)