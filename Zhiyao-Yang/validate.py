import os
import time
import argparse
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import csv
import glob

# 从 ghost_resnet.py 导入模型
from ghost_resnet import resnet50
#from resnet import resnet50

# 设置日志格式
logging.basicConfig(level=logging.INFO)

# AverageMeter 类用于统计平均值
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 计算 top-k 准确率
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

# 训练一个 epoch
def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, args):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start_time = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        acc1, acc5 = accuracy(output, target, (1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0 or batch_idx == len(loader) - 1:
            logging.info(
                'Epoch: [{0}][{1}/{2}]  '
                'Time: {batch_time:.3f}  '
                'Loss: {loss.val:.4f} ({loss.avg:.4f})  '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f})  '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(loader),
                    batch_time=time.time() - start_time,
                    loss=losses, top1=top1, top5=top5))
            start_time = time.time()

    return OrderedDict([('loss', losses.avg), ('top1', top1.avg), ('top5', top5.avg)])

# 验证函数
def validate(model, loader, loss_fn, device, log_suffix=''):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            input, target = input.to(device), target.to(device)

            output = model(input)
            loss = loss_fn(output, target)

            acc1, acc5 = accuracy(output, target, (1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
                logging.info(
                    'Test{}: [{}/{}]  '
                    'Time: {:.3f}  '
                    'Loss: {:.4f} ({:.4f})  '
                    'Acc@1: {:.3f} ({:.3f})  '
                    'Acc@5: {:.3f} ({:.3f})'.format(
                        log_suffix, batch_idx, len(loader),
                        time.time() - end,
                        loss.item(), losses.avg,
                        acc1.item(), top1.avg,
                        acc5.item(), top5.avg))
                end = time.time()

    return OrderedDict([('loss', losses.avg), ('top1', top1.avg), ('top5', top5.avg)])


def main():
    parser = argparse.ArgumentParser(description='Train and Validate Ghost-ResNet on CIFAR-100')
    parser.add_argument('--data', type=str, default='D:\\DeepLearningClass/data/cifar100', help='dataset root')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--num-gpu', default=1, type=int, help='Number of GPUS to use')
    parser.add_argument('--width', type=float, default=4.0, help='Width ratio (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (optional)')
    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据增强和标准化
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 加载原始训练集
    train_dataset_full = datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)

    # 划分训练集和验证集（例如 90% 训练，10% 验证）
    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    # 注意：验证集使用与训练集不同的 transform（通常不进行数据增强）
    # 所以我们可以重新定义 val_dataset 的 transform 为 transform_test
    val_dataset.dataset.transform = transform_test  # 把 transform 改为测试时的 transform（无数据增强）

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # 加载测试集
    test_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    model = resnet50(num_classes=100, s=args.width, d=3)
    #model = resnet50(num_classes=100)

    if args.resume:
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model 'Ghost-ResNet'")

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    print("设备为:",device)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0

    # 创建 results/run_x 文件夹
    result_base_dir = 'results'
    os.makedirs(result_base_dir, exist_ok=True)

    # 获取已有 run_x 文件夹编号
    existing_runs = glob.glob(os.path.join(result_base_dir, 'run_*'))
    existing_ids = [int(x.split('_')[-1]) for x in existing_runs if x.split('_')[-1].isdigit()]
    next_run_id = max(existing_ids, default=-1) + 1  # 如果没有旧文件夹，则从 0 开始
    result_dir = os.path.join(result_base_dir, f'run_{next_run_id}')
    os.makedirs(result_dir, exist_ok=True)

    print(f"Results will be saved to: {result_dir}")

    # 初始化记录器
    train_losses = []
    train_top1 = []
    val_losses = []
    val_top1 = []
    epochs_list = []

    for epoch in range(args.epochs):
        # 训练一个 epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        scheduler.step()

        # 在验证集上评估
        val_metrics = validate(model, val_loader, criterion, device, log_suffix=' (Val)')

        # 打印 summary
        logging.info('Epoch Summary: Train Loss: {:.4f}, Train Acc@1: {:.3f}, Val Loss: {:.4f}, Val Acc@1: {:.3f}'.format(
            train_metrics['loss'], train_metrics['top1'],
            val_metrics['loss'], val_metrics['top1']))

        # 保存最佳模型（基于验证集）
        is_best = val_metrics['top1'] > best_acc
        best_acc = max(val_metrics['top1'], best_acc)
        if is_best:
            torch.save({'state_dict': model.state_dict()}, 'best_ghost_resnet_cifar100.pth')

        # 记录当前 epoch 的指标
        epochs_list.append(epoch + 1)
        train_losses.append(train_metrics['loss'])
        train_top1.append(train_metrics['top1'])
        val_losses.append(val_metrics['loss'])
        val_top1.append(val_metrics['top1'])

    # 最终训练完成后，在测试集上评估一次
    logging.info("Final evaluation on test set...")
    test_metrics = validate(model, test_loader, criterion, device, log_suffix=' (Test)')
    logging.info('Test Results: Loss: {:.4f}, Acc@1: {:.3f}, Acc@5: {:.3f}'.format(
        test_metrics['loss'], test_metrics['top1'], test_metrics['top5']))

    # 保存为 CSV 文件
    csv_path = os.path.join(result_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        for i in range(len(epochs_list)):
            writer.writerow([
                epochs_list[i],
                train_losses[i],
                train_top1[i],
                val_losses[i],
                val_top1[i]
            ])

    # 绘制 loss 曲线
    plt.figure()
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    plt.close()

    # 绘制 accuracy 曲线
    plt.figure()
    plt.plot(epochs_list, train_top1, label='Train Acc@1')
    plt.plot(epochs_list, val_top1, label='Val Acc@1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'accuracy_curve.png'))
    plt.close()

if __name__ == '__main__':
    main()