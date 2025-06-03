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
from torchvision import transforms, datasets

# 从 ghost_resnet.py 导入模型
from ghost_resnet import resnet50

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


        # print("Output min/max:", output.min().item(), output.max().item())  # 👈 加在这里



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
    parser.add_argument('--data', type=str, default='./data/cifar100', help='dataset root')
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

    # 加载数据集
    train_dataset = datasets.CIFAR100(root=args.data, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=args.data, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)


    # for input, target in train_loader:
    #     print(input.shape, input.min(), input.max())   # 应该是 torch.Size([128, 3, 32, 32])，值在 [0, 1] 或 [-x, x]
    #     print(target.min(), target.max())              # 应该是 0~99
    #     break




    # 创建模型
    model = resnet50(num_classes=100, s=args.width, d=3)

    if args.resume:
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model 'Ghost-ResNet'")

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0

    # 开始训练
    for epoch in range(args.epochs):
        # 训练一个 epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        scheduler.step()

        # 验证
        test_metrics = validate(model, test_loader, criterion, device)

        # 打印总结
        logging.info('Epoch Summary: Train Loss: {:.4f}, Train Acc@1: {:.3f}, Test Loss: {:.4f}, Test Acc@1: {:.3f}, Test Acc@5: {:.3f}'.format(
            train_metrics['loss'], train_metrics['top1'],
            test_metrics['loss'], test_metrics['top1'], test_metrics['top5']))

        # 保存最佳模型
        is_best = test_metrics['top1'] > best_acc
        best_acc = max(test_metrics['top1'], best_acc)
        if is_best:
            torch.save({'state_dict': model.state_dict()}, 'best_ghost_resnet_cifar100.pth')

    print('Training finished.')
    print(f'Best Top-1 Accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()