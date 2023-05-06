import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse

import config
from ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from data.dataset import ICDARDataset



# dataset_download:https://rrc.cvc.uab.es/?ch=8&com=downloads
random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

#epoch本来是30 训练时间太长了 改成了2
epochs = 3
lr = 1e-3       #学习率0.001
#lr = 1e-2
resume_epoch = 0

#保存训练完的权值文件：02d整数两位数字，.4f浮点数保留四位数字

def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(config.checkpoints_dir,
                              f'v3_ctpn_ep{epoch:02d}_'
                              f'{loss_cls:.4f}_{loss_regr:.4f}_{loss:.4f}.{ext}')

    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print('fail to save to {}'.format(check_path))
    print('saving to {}'.format(check_path))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)    #使用正态分布初始化卷积核权重，均值为 0，标准差为 0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)    #则将批归一化的权重初始化为 1，偏置项初始化为 0。
        m.bias.data.fill_(0)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoints_weight = config.pretrained_weights                  #以前训练的权重文件
    print('exist pretrained ',os.path.exists(checkpoints_weight))
    if os.path.exists(checkpoints_weight):
        pretrained = False

    dataset = ICDARDataset(config.icdar17_mlt_img_dir, config.icdar17_mlt_gt_dir)       #加载数据 img和标签进行网路训练
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    model = CTPN_Model()
    model.to(device)
    
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))     #使用之前的权重文件继续进行训练
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weights_init)

    params_to_uodate = model.parameters()   #是需要更新的模型参数，一般情况下是所有参数。
    optimizer = optim.SGD(params_to_uodate, lr=lr, momentum=0.9)
    
    critetion_cls = RPN_CLS_Loss(device)    #分类损失和回归损失都使用之前定义的 RPN_CLS_Loss 和 RPN_REGR_Loss 函数计算。其中，device 参数表示将损失函数计算在 CPU 还是 GPU 上。
    critetion_regr = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100    #在每次 epoch 完成后，如果当前总体误差小于之前的最佳误差，则将当前模型保存为最佳模型。
    best_loss = 100
    best_model = None
    epochs += resume_epoch  #表示继续从上一次训练结束的位置开始训练，而不是从头开始
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   #scheduler 用于控制学习率的变化，在每个 step_size（这里设为 10）个 epoch 后将学习率乘以 gamma（这里设为 0.1），以加快模型收敛速度。
    
    for epoch in range(resume_epoch+1, epochs):
        print(f'Epoch {epoch}/{epochs}')
        print('#'*50)
        epoch_size = len(dataset) // 1
        model.train()           #并调用模型的 train() 方法以将其设置为训练模式。
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)   #调用调度器的 step() 方法，传入当前 epoch 的编号作为参数
    
        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):  #这段代码是在每个 epoch 中遍历 DataLoader 中的所有 batch
            # print(imgs.shape) #循环中 enumerate(dataloader) 会从给定的数据加载器中返回当前 batch 的索引以及对应的图像、类别和回归值。
            # 将这些张量移动到 GPU 设备上以便并行处理。
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()   #调用优化器的 zero_grad() 方法，清零所有参数的梯度。
    
            out_cls, out_regr = model(imgs)
            loss_cls = critetion_cls(out_cls, clss)
            loss_regr = critetion_regr(out_regr, regrs)
    
            loss = loss_cls + loss_regr  # total loss

            #释放内存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loss.backward()     #计算损失函数并执行反向传播
            optimizer.step()    # 更新参数 例如，SGD优化器使用以下公式更新参数：参数 = 参数 - 学习率 * 参数的梯度
    
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i+1
    
            print(f'Ep:{epoch}/{epochs-1}--'
                  f'Batch:{batch_i}/{epoch_size}\n'
                  f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
                  f'Epoch: loss_cls:{epoch_loss_cls/mmp:.4f}--loss_regr:{epoch_loss_regr/mmp:.4f}--'
                  f'loss:{epoch_loss/mmp:.4f}\n')
    
        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print(f'Epoch:{epoch}--{epoch_loss_cls:.4f}--{epoch_loss_regr:.4f}--{epoch_loss:.4f}')
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:    #确保保存表现最佳的模型版本，以供评估或进一步训练使用。
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(),
                             'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

