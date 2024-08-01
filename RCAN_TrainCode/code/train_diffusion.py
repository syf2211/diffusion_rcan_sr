from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers import LMSDiscreteScheduler
import torch
import torch.nn.functional as F
from PIL import Image
from mydataloader import MyDataset
from torch.utils.data import DataLoader
from my_unet import PromptSRUnet
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torchvision import transforms
from tifffile import imwrite
def train(model,dataloader,optimizer,device,epoch,scheduler):
    model.train()   
    epoch_loss = 0
    for iteration, batch in enumerate(dataloader, 1):
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        noise=torch.randn(batch['lq'].shape).to(device)
        bsz=lq.shape[0]
        timesteps=torch.randint(0,scheduler.config.num_train_timesteps,(bsz,)).to(device)
        timesteps=timesteps.long()
        noisy_image=scheduler.add_noise(gt,noise,timesteps)
        target=noise
        optimizer.zero_grad()
        noise_pred=model(lq,noisy_image,timesteps)
        loss = F.l1_loss(noise_pred, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if iteration % 10 == 0:
            print(f"Epoch [{epoch}], Iteration [{iteration}], Loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
def test(lq,gt,model,device,scheduler,inference_steps):
    model.eval()
    
    latents=torch.randn(lq.shape).to(device)
    lq=lq.to(device)
    scheduler.set_timesteps(inference_steps,device)
    for t in scheduler.timesteps:
        latents_model_input=latents
        t=t.unsqueeze(0).to('cpu')
        with torch.no_grad():
            noise_pred=model(lq,latents_model_input,t)
        
        latents=scheduler.step(noise_pred,t,latents).prev_sample
    val_loss = F.l1_loss(noise_pred, gt)
    print(f"Validation Loss: {val_loss.item():.4f}")
    return latents
def validate(model,dataloader,scheduler,device,inference_steps,epoch):
    model.eval()
    sum_psnr=0
    sum_ssim=0
    count=0
    Center_crop = transforms.CenterCrop([256,256])
    for  batch in dataloader:
        count+=1
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        lq=Center_crop(lq)
        gt=Center_crop(gt)
        #标准化
        gt=(gt-gt.mean())/gt.std()  
        lq=(lq-lq.mean())/lq.std()
        sr=test(lq,gt,model,device,scheduler,inference_steps)
        sr_numpy=sr.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
        gt_numpy=gt.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
        data_range=gt_numpy.max()-gt_numpy.min()
        print(f'data_range: {data_range}')
        #归一化
        sr_numpy=(sr_numpy-sr_numpy.min())/(sr_numpy.max()-sr_numpy.min())
        gt_numpy=(gt_numpy-gt_numpy.min())/(gt_numpy.max()-gt_numpy.min())
        psnr_value = PSNR(sr_numpy, gt_numpy,data_range=data_range)
        ssim_value = SSIM(sr_numpy, gt_numpy,data_range=data_range)
        print(f'PSNR: {psnr_value}')
        print(f'SSIM: {ssim_value}')
        sum_psnr+=psnr_value
        sum_ssim+=ssim_value
        sr_path='sr_val_diff'
        gt_path='gt_val_diff'
        lq_path='lq_val_diff'
        import os
        if not os.path.exists(sr_path):
            os.makedirs(sr_path)
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        if not os.path.exists(lq_path):
            os.makedirs(lq_path)
        imwrite(f"{sr_path}/sr_{count}_{ssim_value:.2f}_{psnr_value:.2f}_epoch{epoch}.tif", sr_numpy)
        imwrite(f"{gt_path}/gt_{count}.tif", gt_numpy)
        imwrite(f"{lq_path}/lq_{count}.tif", lq.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1))
    avg_psnr=sum_psnr/len(dataloader)
    avg_ssim=sum_ssim/len(dataloader)
    print(f"Epoch [{epoch}] Average PSNR: {avg_psnr:.4f}")
    print(f"Epoch [{epoch}] Average SSIM: {avg_ssim:.4f}")
    return avg_psnr
def main(num_train_timesteps=2000,num_inference_steps=100):
    # 设置模型的超参数
    model=PromptSRUnet(in_channel=2,out_channel=1,inner_channel=32,norm_groups=16,channel_mults= [1, 2, 4, 8],attn_res= [],res_blocks= 2
  ,dropout= 0.2,
  image_size= 256,
  attention_levels= [2, 3],
  n_heads= 16,
  tf_layers= 1,
  d_cond=768)
    opt = {
    'train_dataroot': 'train_diff',  
    'val_dataroot': 'val_diff',          
    'structure':['PKMO','F-actin','ER','Lyso','Ensconsin','TOMM20','PHB2','CCPs'], 
    'structure_selection': [1,0,0,0,0,0,0,0],            
    'snr': ['150', '200', '300', '500', '900'],                  
    'gt_size': 256,                           
}
    train_dataset = MyDataset(opt, is_train=True)
    val_dataset = MyDataset(opt, is_train=False)
    train_dataloader=DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)
    val_dataloader=DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    #设置Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6,betas=(0.9, 0.99))
    #设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #设置scheduler
    scheduler = DDPMScheduler(beta_schedule="scaled_linear", num_train_timesteps=num_train_timesteps)
    scheduler.register_to_config(prediction_type='epsilon')
    #开始训练
    num_epochs = 1000
    best_psnr=float('-inf')
    for epoch in range(num_epochs):
        # validate(model,val_dataloader,scheduler,device,num_inference_steps,epoch)
        train(model,train_dataloader,optimizer,device,epoch,scheduler)
        if (epoch+1) % 10 == 0:
            # psnr=validate(model,val_dataloader,scheduler,device,num_inference_steps,epoch)
            torch.save(model, "best_model_diffusion_norm.pth")
            psnr=validate(model,val_dataloader,scheduler,device,num_inference_steps,epoch)
            if psnr>best_psnr:
                best_psnr=psnr
                #保存整个模型
                torch.save(model, "best_model_norm.pth")
    
if __name__ == '__main__':
    num_train_timesteps=2000
    num_inference_steps=100
    main(num_train_timesteps,num_inference_steps)