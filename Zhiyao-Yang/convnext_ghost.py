import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from timm.layers import trunc_normal_, DropPath
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import math
import random

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(42)


save_dir = './ghost_newenhanced_scratch_checkpoints'
os.makedirs(save_dir, exist_ok=True)


curve_path = os.path.join(save_dir, 'training_curves.png')
final_curve_path = os.path.join(save_dir, 'final_training_curves.png')

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        

        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
      
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
      
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
class LayerNorm(nn.Module):
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

import math

class ECABlock(nn.Module):
    
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
       
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1  
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
       
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
       
        identity = x
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return identity + x * y

# ghost模块，用于减少参数量
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dw_size=3, ratio=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        
        super(GhostModule, self).__init__()  # 不传任何参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dw_size = dw_size
        self.ratio = ratio
        self.stride = stride
        self.padding = padding
        self.groups = groups
        # super(GhostModule, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = None
        self.ratio = ratio
        self.dw_size = dw_size
        self.dw_dilation = (dw_size - 1) // 2
        self.init_channels = math.ceil(out_channels / ratio) # 主特征数量
        self.new_channels = int(self.init_channels * (ratio - 1)) # ghost 特征数量
        
        self.conv1 = nn.Conv2d(self.in_channels, self.init_channels, kernel_size, self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, padding=int(self.dw_size/2), groups=self.init_channels)
        
        
        self.weight1 = nn.Parameter(torch.Tensor(self.init_channels, self.in_channels, kernel_size, kernel_size))
        self.bn1 = nn.BatchNorm2d(self.init_channels)
        if self.new_channels > 0:
            self.weight2 = nn.Parameter(torch.Tensor(self.new_channels, 1, self.dw_size, self.dw_size))
            self.bn2 = nn.BatchNorm2d(self.out_channels - self.init_channels)
        
        if bias:
            self.bias =nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        if self.new_channels > 0:
            nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input): 
        x1 = self.conv1(input)
        if self.new_channels == 0:
            return x1
        x2 = self.conv2(x1)
        x2 = x2[:, :self.out_channels - self.init_channels, :, :]
        x = torch.cat([x1, x2], 1)
        return x

class Block(nn.Module):
    """增强的ConvNeXt Block，引入ECA模块"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 在深度可分离卷积后、LayerNorm前插入ECA模块
        # self.eca = ECABlock(dim)
        
        self.norm = LayerNorm(dim, eps=1e-6)

        # 替换为ghost模块
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv1 = GhostModule(dim, 4 * dim, kernel_size=1, dw_size=3, ratio=4, bias=False)

        self.act = nn.GELU()

        # 替换为ghost模块
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = GhostModule(4 * dim, dim , kernel_size=1, dw_size=3, ratio=4, bias=False)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    # def forward(self, x):
    #     input = x
    #     print("Input shape:", input.shape)

    #     x = self.dwconv(x)
    #     print("After dwconv:", x.shape)

    #     x = self.eca(x)
    #     print("After ECA:", x.shape)

    #     x = x.permute(0, 2, 3, 1)
    #     print("After NHWC permute:", x.shape)

    #     x = self.norm(x)
    #     print("After norm:", x.shape)

    #     x = x.permute(0, 3, 1, 2)
    #     print("Back to NCHW before pwconv1:", x.shape)

    #     x = self.pwconv1(x)
    #     print("After pwconv1:", x.shape)

    #     x = self.act(x)
    #     x = self.pwconv2(x)
    #     print("After pwconv2:", x.shape)

    #     if self.gamma is not None:
    #         x = x.permute(0, 2, 3, 1)
    #         x = self.gamma * x
    #         x = x.permute(0, 3, 1, 2)
    #         print("After gamma scaling:", x.shape)

    #     print("Final output shape:", x.shape)
    #     print("Residual input shape:", input.shape)

    #     x = input + self.drop_path(x)
    #     return x


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # 引入ECA注意力机制
        # x = self.eca(x)
        # 继续原有的处理流程
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)

        # 添加一个转换逻辑，兼容ghost模块
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1)  # 回 NHWC 才能做 channel-wise scaling
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # 回 NCHW
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.permute(0, 3, 1, 2) 

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=100, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.2,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) 

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnext_tiny_cifar100_with_eca(pretrained=False):

    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     num_classes=100, drop_path_rate=0.2)
    return model
def get_cifar100_loaders(batch_size=64, num_workers=2):

    

    train_transform = transforms.Compose([
        transforms.Resize(224),  # ConvNeXt期望224x224输入
        transforms.RandomCrop(224, padding=28),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomRotation(degrees=15),   # 随机旋转 ±15度
        transforms.ColorJitter(
            brightness=0.2,   # 亮度变化
            contrast=0.2,     # 对比度变化
            saturation=0.2,   # 饱和度变化
            hue=0.1          # 色调变化
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.2),  # 高斯模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 随机擦除
    ])
    
    # 验证集只做基本预处理，不使用数据增强
    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    # 加载CIFAR100数据集
    train_dataset = CIFAR100(root='D:\\DeepLearningClass/data/cifar100', train=True, download=True, transform=train_transform)
    valid_dataset = CIFAR100(root='D:\\DeepLearningClass/data/cifar100', train=False, download=True, transform=valid_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, valid_loader


train_loader, valid_loader = get_cifar100_loaders(batch_size=64)
print(f"数据集大小 - 训练集: {len(train_loader.dataset)}, 验证集: {len(valid_loader.dataset)}")
def train_one_epoch(model, train_loader, criterion, optimizer, device, mixup=None, cutmix=None, use_mixup_prob=0.8):
    """训练一个epoch - 支持Mixup和CutMix"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='训练中')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 随机选择是否使用数据增强
        use_mixup = random.random() < use_mixup_prob
        use_cutmix = random.random() < 0.5  # 50%概率在mixup和cutmix之间选择
        
        if use_mixup:
            if use_cutmix and cutmix is not None:
                inputs, targets_a, targets_b, lam = cutmix(inputs, targets)
            elif mixup is not None:
                inputs, targets_a, targets_b, lam = mixup(inputs, targets)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
        else:
            targets_a, targets_b, lam = targets, targets, 1.0
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_mixup and (mixup is not None or cutmix is not None):
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if use_mixup:
            
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': train_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    return train_loss/len(train_loader), 100.*correct/total

def validate(model, valid_loader, criterion, device):
    """验证函数"""
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc='验证中')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': valid_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
    
    return valid_loss/len(valid_loader), 100.*correct/total
def plot_enhanced_training_curves(train_losses, valid_losses, train_accs, valid_accs, learning_rates, save_path=curve_path):
    """绘制训练曲线并保存"""
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(valid_losses, label='valid_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='train_accuracy')
    plt.plot(valid_accs, label='valid_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.title('Accuracy Curve')

    
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
# 设置训练参数
lr = 5e-4                 # 基础学习率
min_lr = 1e-6            # 最小学习率
batch_size = 64          # 批次大小  
num_epochs = 120         # 训练轮次
warmup_epochs = 10       # 预热轮次
weight_decay = 0.05      # 权重衰减
label_smoothing = 0.1    # 标签平滑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"使用设备: {device}")
model = convnext_tiny_cifar100_with_eca(pretrained=False)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, lr, min_lr)


mixup = MixUp(alpha=0.2)
cutmix = CutMix(alpha=1.0)

print(f"\n训练配置:")
print(f"- 基础学习率: {lr}")
print(f"- 最小学习率: {min_lr}")
print(f"- 预热轮次: {warmup_epochs}")
print(f"- 权重衰减: {weight_decay}")
print(f"- 标签平滑: {label_smoothing}")
print(f"- Stochastic Depth: 0.2")
print(f"- Mixup alpha: 0.2")
print(f"- CutMix alpha: 1.0")
# 训练循环
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []
learning_rates = [] 
best_acc = 0

print(f"\n开始训练模型，总共{num_epochs}轮...")

for epoch in range(num_epochs):
    print(f"\n=== 轮次: {epoch+1}/{num_epochs} ===")
    
    # 更新学习率
    current_lr = lr_scheduler.step(epoch)
    learning_rates.append(current_lr)
    print(f"当前学习率: {current_lr:.6f}")
    
    # 训练 - 使用Mixup和CutMix
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, 
        mixup=mixup, cutmix=cutmix, use_mixup_prob=0.8  # 80%概率使用数据增强
    )
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 验证
    valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
    valid_losses.append(valid_loss)
    valid_accs.append(valid_acc)
    
    print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"验证 - Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%")
    
    # 保存模型
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': valid_acc,
            'config': {
                'lr': lr,
                'min_lr': min_lr,
                'weight_decay': weight_decay,
                'label_smoothing': label_smoothing,
                'warmup_epochs': warmup_epochs,
                'drop_path_rate': 0.2,
                'mixup_alpha': 0.2,
                'cutmix_alpha': 1.0
            }
        }, os.path.join(save_dir, 'best_enhanced_model.pth'))
        print(f"保存最佳模型，准确率: {valid_acc:.2f}%")
    
    # 每10轮绘制一次训练曲线
    if (epoch + 1) % 10 == 0 or epoch == 0:
        plot_enhanced_training_curves(train_losses, valid_losses, train_accs, valid_accs, learning_rates, curve_path)

# 最后绘制完整训练曲线
plot_enhanced_training_curves(train_losses, valid_losses, train_accs, valid_accs, learning_rates, final_curve_path)
print(f"\n 训练完成! 最佳验证准确率: {best_acc:.2f}%")
# 加载模型测试
print("加载模型进行测试...")
best_model = convnext_tiny_cifar100_with_eca(pretrained=False)
checkpoint = torch.load(os.path.join(save_dir, 'best_enhanced_model.pth'), map_location=device)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model = best_model.to(device)

test_criterion = nn.CrossEntropyLoss()
test_loss, test_acc = validate(best_model, valid_loader, test_criterion, device)
print(f"\n 模型在测试集上的准确率: {test_acc:.2f}%")
print(f"训练配置信息:")
for key, value in checkpoint['config'].items():
    print(f"  - {key}: {value}")