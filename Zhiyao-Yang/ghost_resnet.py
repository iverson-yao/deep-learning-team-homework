import math
import torch
import torch.nn as nn


class GhostModule(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dw_size=3, ratio=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GhostModule, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
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