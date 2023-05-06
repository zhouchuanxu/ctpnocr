#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:01
#
# @Author: Greg Gao(laygin)
#'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config

class RPN_REGR_Loss(nn.Module): #这是一个 RPN 回归损失函数的实现。该函数采用 smooth L1 损失计算预测值和目标值之间的差异，并将其应用于分类器中的正样本上。
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        '''
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        '''
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            # apply regression to positive sample
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff<1.0/self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device) #其中，sigma 是平滑参数，cls 是分类结果的张量（表示每个锚点框是否为物体或背景），regr 是回归结果的张量（表示每个锚点框的偏移量）。函数首先获取所有正样本的索引 regr_keep，然后从 regr 中提取出这些正样本的真实值 regr_true 和预测值 regr_pred，计算它们之间的差异 diff。接着，使用 smooth L1 损失计算损失值，最后返回平均值。


class RPN_CLS_Loss(nn.Module):
    def __init__(self,device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')
        # self.L_regr = nn.SmoothL1Loss()
        # self.L_refi = nn.SmoothL1Loss()
        self.pos_neg_ratio = 3

    def forward(self, input, target):
        if config.OHEM:
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0

            # print(len((cls_gt == 0).nonzero()),len((cls_gt == 1).nonzero()))

            if len((cls_gt == 1).nonzero())!=0:       # avoid num of pos sample is 0
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                gt_pos = cls_gt[cls_pos].long()
                cls_pred_pos = input[0][cls_pos]
                # print(cls_pred_pos.shape)
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)

            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            gt_neg = cls_gt[cls_neg].long()
            cls_pred_neg = input[0][cls_neg]

            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM-num_pos))
            loss_cls = loss_pos_sum+loss_neg_topK.sum()
            loss_cls = loss_cls/config.RPN_TOTAL_NUM
            return loss_cls.to(self.device)
        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = input[0][cls_keep]
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1),
                              cls_true)  # original is sparse_softmax_cross_entropy_with_logits
            # loss = nn.BCEWithLogitsLoss()(cls_pred[:,0], cls_true.float())  # 18-12-8
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
            return loss.to(self.device)


class basic_conv(nn.Module):
    def __init__(self,
                 in_planes, #表示输入通道数
                 out_planes,    # 表示输出通道数，
                 kernel_size,   #表示卷积核大小
                 stride=1,  #e 表示步长
                 padding=0, #padding 表示填充值
                 dilation=1,    #dilation 表示卷积核膨胀率
                 groups=1,  #示分组卷积的组数
                 relu=True, #表示是否使用 ReLU 激活函数
                 bn=True,   #bn 表示是否使用批归一化
                 bias=True):    #bias 表示是否使用偏置项。
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)     #VGG16 model加载进去
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)  # block5_conv3 output
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)                  #加了一个卷积层
        self.brnn = nn.GRU(512,128, bidirectional=True, batch_first=True)   #加了一个双向rnn 128x2个特征
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)      #全连接层256个特征接受
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)#分类十个不同框进行 20个背景还是不是背景
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)#y值和h怎么进行偏移量

    def forward(self, x):
        #print ('1:',x.size())
        x = self.base_layers(x) #将输入张量通过卷积基础网络进行特征提取。
        #print ('2:',x.size())
        # rpn
        x = self.rpn(x)    #[b, c, h, w]    #在基础网络的输出上执行区域建议网络（RPN）计算。
        #print ('3:',x.size())
        x1 = x.permute(0,2,3,1).contiguous()  # channels last   [b, h, w, c]    将 x 的维度转换为通道数在最后一维的形式。
        #print ('4:',x1.size())
        b = x1.size()  # b, h, w, c
        x1 = x1.view(b[0]*b[1], b[2], b[3]) #将张量展平，并将其重新调整为适合双向 LSTM 输入的形状。
        #print ('5:',x1.size())
        x2, _ = self.brnn(x1)   #将经过展平和重塑后的张量传递给一个双向 LSTM。
        #print ('6:',x2.size())
        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256]) 将 LSTM 输出的张量重塑为与 RPN 输入张量相同的大小。
        #print ('7:',x3.size())
        x3 = x3.permute(0,3,1,2).contiguous()  # channels first [b, c, h, w]  x3 的维度恢复为通道数在第一维的形式。
        #print ('8:',x3.size())
        x3 = self.lstm_fc(x3) #将 LSTM 输出的张量传递给一个全连接层。
        #print ('9:',x3.size())
        x = x3

        cls = self.rpn_class(x)     #获取表示每个锚点框是否为物体或背景的概率的张量。
        #print ('10:',cls.size())
        regr = self.rpn_regress(x) #获取表示每个锚点框偏移量的张量。
        #print ('11:',regr.size())
        cls = cls.permute(0,2,3,1).contiguous()     #将 cls 的维度恢复为通道数在最后一维的形式。
        regr = regr.permute(0,2,3,1).contiguous()   #将 regr 的维度恢复为通道数在最后一维的形式。

        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)  #将 cls 张量展平，并将其重新调整为每个锚点框输出两个值（物体或背景）的形状。
        #print ('12:',cls.size())
        regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2) #将 regr 张量展平，并将其重新调整为每个锚点框输出两个值（偏移量）的形状。
        #print ('13:',regr.size())

        return cls, regr
