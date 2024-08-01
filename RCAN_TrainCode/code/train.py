#导入dataloader等库
from  torch.utils.data import DataLoader
import torch
# from data import common
# import utility
# import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mydataloader import MyDataset
from my_rcan import RCAN
from torch.functional import F
from torch.optim import Adam
from metric import psnr, ssim
#tiff文件读取库
from PIL import Image
import numpy as np
#imwrite库
from tifffile import imwrite
#导入transform
from torchvision import transforms
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import math
opt = {
    'train_dataroot': 'train_diff',  
    'val_dataroot': 'val_diff',          
    'structure':['PKMO','F-actin','ER','Lyso','Ensconsin','TOMM20','PHB2','CCPs'], 
    'structure_selection': [1,0,0,0,0,0,0,0],            
    'snr': ['150', '200', '300', '500', '900'],                  
    'gt_size': 256,                           
}
train_set = MyDataset(opt)
train_loader = DataLoader(train_set, batch_size=2, shuffle=False, num_workers=4)

def train(epoch, model, optimizer, device):
    epoch_loss = 0
    model.train()  # 将模型设置为训练模式
    
    for iteration, batch in enumerate(train_loader, 1):
        lq = batch['lq'].to(device)  # 获取数据并转移到 GPU
        gt = batch['gt'].to(device)
        
        optimizer.zero_grad()  # 清除旧的梯度
        sr = model(lq)  # 进行前向传播
        
        loss = F.l1_loss(sr, gt)  # 计算损失
        loss.backward()  # 进行反向传播
        optimizer.step()  # 更新优化器
        
        epoch_loss += loss.item()  # 累计损失
        
        if iteration % 10 == 0:  # 每 10 个 batch 打印一次
            print(f"Epoch [{epoch}], Iteration [{iteration}], Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
def validate(model, device,epoch):
    model.eval()  # 将模型设置为验证模式
    val_set = MyDataset(opt, is_train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        psnr_val_total = 0
        ssim_val_total = 0
        count = 0
        Centercrop=transforms.CenterCrop([256,256])
        for batch in val_loader:
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)
            #中心裁剪
            lq=Centercrop(lq)
            gt=Centercrop(gt)
            # 归一化
            # print("before normalization")
            # print(gt)
            # # 保存一下原图像
            # gt_before=gt.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
            # gt_before=gt_before
            # imwrite(f"gt_before_{count}.tiff", gt_before)
            
            gt_max=gt.max().item()
            gt_min=gt.min().item()
            lq_max=lq.max().item()
            lq_min=lq.min().item()
            lq=(lq-lq.min())/(lq.max()-lq.min())
            gt=(gt-gt.min())/(gt.max()-gt.min())
            sr = model(lq)
            loss = F.l1_loss(sr, gt)
            #将loss写入文件
            with open('val_loss.txt', 'a') as f:
                f.write(f"Validation Loss: {loss.item():.4f}\n")
            # print(sr.shape)
            # 保存验证集的结果
            #先将tensor转换为numpy数组，再保存为图片（16bit tiff）imwrite
            
            sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
            gt = gt.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
            lq = lq.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
            # print(np.sum(np.abs(sr-gt)))
            ssim_val = SSIM(sr, gt, data_range=1)
            psnr_val = psnr(sr, gt)
            print(f"Validation Loss: {loss.item():.4f}")
            print(f"Validation PSNR: {psnr_val:.2f}")
            print(f"Validation SSIM: {ssim_val:.4f}")
            psnr_val_total += psnr_val
            ssim_val_total += ssim_val
            # breakpoint()
            count += 1
            # sr = (sr * (gt_max - gt_min) + gt_min)
            # gt = (gt * (gt_max - gt_min) + gt_min)
            # lq=(lq * (lq_max - lq_min) + lq_min)

            # print("after denormalization")
            # print(gt)
            print(np.sum(np.abs(sr-gt))/65535)
            # breakpoint()
            # ssim保留两位小数
            sr_path='sr_val4'
            gt_path='gt_val4'
            lq_path='lq_val4'
            import os
            if not os.path.exists(sr_path):
                os.makedirs(sr_path)
            if not os.path.exists(gt_path):
                os.makedirs(gt_path)
            if not os.path.exists(lq_path):
                os.makedirs(lq_path)
            imwrite(f"{sr_path}/sr_{count}_{ssim_val:.2f}_{psnr_val:.2f}_epoch{epoch}.tif", sr)
            imwrite(f"{gt_path}/gt_{count}.tif", gt)
            imwrite(f"{lq_path}/lq_{count}.tif", lq)
        avg_psnr = psnr_val_total / count
        avg_ssim = ssim_val_total / count
        #将指标写入文件
        with open('val_result.txt', 'a') as f:
            f.write(f"Epoch [{epoch}] Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Epoch [{epoch}] Average SSIM: {avg_ssim:.4f}\n")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
    
    return avg_psnr  # 返回平均 PSNR
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RCAN().to(device)  # 将模型转移到 GPU
    weights = torch.load("best_model.pth",weights_only=True)
    model.load_state_dict(weights)
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0 )  # 初始化优化器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)  # 初始化调度器
    num_epochs = 1000  # 设定训练的 epoch 数量
    best_psnr = float('-inf')  # 初始化最佳 PSNR 为负无穷
    save_path = "best_model.pth"  # 最佳模型保存路径
    
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, optimizer, device)
        if epoch % 10 == 0:
            avg_psnr = validate(model, device,epoch)  # 验证模型并获取平均 PSNR
            
            # 更新最佳模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path} with PSNR: {best_psnr:.2f}")
            
            # 更新学习率
            scheduler.step(avg_psnr)
if __name__ == '__main__':
    main()