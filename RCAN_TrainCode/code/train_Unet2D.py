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
import time
#引入预训练的clip编码器
from transformers import CLIPTokenizer, CLIPTextModel
def train(model,dataloader,optimizer,device,epoch,scheduler,tokenizer,text_encoder):
    model.train()   
    epoch_loss = 0
    for iteration, batch in enumerate(dataloader, 1):
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        text=batch['text']
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            text_features = text_encoder(**inputs).last_hidden_state
        text_features=text_features.to(device)
        noise=torch.randn(batch['lq'].shape).to(device)
        bsz=lq.shape[0]
        timesteps=torch.randint(0,scheduler.config.num_train_timesteps,(bsz,)).to(device)
        timesteps=timesteps.long()
        noisy_image=scheduler.add_noise(gt,noise,timesteps)
        target=noise
        optimizer.zero_grad()
        input=torch.cat([lq,noisy_image],dim=1)
        noise_pred=model(input,timesteps,text_features)
        loss = F.l1_loss(noise_pred, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if iteration % 10 == 0:
            print(f"Epoch [{epoch}], Iteration [{iteration}], Loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")
def test(lq,gt,text,model,device,scheduler,inference_steps,tokenizer,text_encoder):
    model.eval()
    latents=torch.randn(lq.shape).to(device)
    lq=lq.to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_features = text_encoder(**inputs).last_hidden_state
    text_features=text_features.to(device)
    scheduler.set_timesteps(inference_steps,device)
    for t in scheduler.timesteps:
        latents_model_input=latents
        t=t.unsqueeze(0).to('cpu')
        input=torch.cat([lq,latents_model_input],dim=1)
        with torch.no_grad():
            noise_pred=model(input,t,text_features)        
        latents=scheduler.step(noise_pred,t,latents).prev_sample
    val_loss = F.l1_loss(noise_pred, gt)
    print(f"Validation Loss: {val_loss.item():.4f}")
    return latents
def validate(model,dataloader,scheduler,device,inference_steps,epoch,tokenizer,text_encoder):
    model.eval()
    sum_psnr=0
    sum_ssim=0
    count=0
    Center_crop = transforms.CenterCrop([256,256])
    for batch in dataloader:
        count+=1
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)
        text=batch['text'] 
        lq=Center_crop(lq)
        gt=Center_crop(gt)
        lq=(lq-lq.min())/(lq.max()-lq.min())
        gt=(gt-gt.min())/(gt.max()-gt.min())
        sr=test(lq,gt,text,model,device,scheduler,inference_steps,tokenizer,text_encoder)
        sr_numpy=sr.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
        gt_numpy=gt.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze(-1)
        data_range=gt_numpy.max()-gt_numpy.min()
        print(f'data_range: {data_range}')
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
def main(num_train_timesteps=2000,num_inference_steps=100):
    # 设置模型的超参数
    unet_config = {
    "sample_size": (256, 256),  # 输入输出图像的尺寸为256x256
    "in_channels": 3,  # 输入通道数为3，表示RGB图像
    "out_channels": 3,  # 输出通道数为3，表示RGB图像
    "center_input_sample": False,  # 不需要对输入样本进行中心化
    "flip_sin_to_cos": True,  # 时间嵌入中使用sin到cos的转换
    "freq_shift": 0,  # 不进行频率偏移
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),  # 下采样块类型
    "mid_block_type": "UNetMidBlock2D",  # 中间块类型
    "up_block_types": (
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),  # 上采样块类型
    "only_cross_attention": False,  # 不使用仅交叉注意力
    "block_out_channels": (64, 128, 256, 512),  # 各块的输出通道数
    "layers_per_block": 2,  # 每个块中的层数
    "downsample_padding": 1,  # 下采样卷积的填充
    "mid_block_scale_factor": 1.0,  # 中间块的缩放因子
    "dropout": 0.0,  # 不使用dropout
    "act_fn": "silu",  # 激活函数
    "norm_num_groups": 32,  # 归一化组数
    "norm_eps": 1e-5,  # 归一化的epsilon值
    "cross_attention_dim": 512,  # 交叉注意力的维度
    "transformer_layers_per_block": 1,  # 每个块中的transformer层数
    "reverse_transformer_layers_per_block": None,  # 不使用反向transformer层
    "encoder_hid_dim": 512,  # 编码器隐藏维度
    "encoder_hid_dim_type": "text_image_proj",  # 编码器隐藏维度类型
    "attention_head_dim": 8,  # 注意力头的维度
    "num_attention_heads": None,  # 注意力头的数量
    "dual_cross_attention": False,  # 不使用双交叉注意力
    "use_linear_projection": False,  # 不使用线性投影
    "class_embed_type": None,  # 不使用类嵌入
    "addition_embed_type": None,  # 不使用额外嵌入
    "addition_time_embed_dim": None,  # 不使用额外的时间嵌入维度
    "num_class_embeds": None,  # 不使用类嵌入的数量
    "upcast_attention": False,  # 不使用上采样注意力
    "resnet_time_scale_shift": "default",  # ResNet时间缩放移位配置
    "resnet_skip_time_act": False,  # 不跳过时间激活
    "resnet_out_scale_factor": 1.0,  # ResNet输出缩放因子
    "time_embedding_type": "positional",  # 时间嵌入类型
    "time_embedding_dim": None,  # 时间嵌入维度
    "time_embedding_act_fn": None,  # 时间嵌入的激活函数
    "timestep_post_act": None,  # 时间步的后激活函数
    "time_cond_proj_dim": None,  # 时间条件投影维度
    "conv_in_kernel": 3,  # 输入卷积核大小
    "conv_out_kernel": 3,  # 输出卷积核大小
    "projection_class_embeddings_input_dim": None,  # 不使用类嵌入输入维度
    "attention_type": "default",  # 注意力类型
    "class_embeddings_concat": False,  # 不连接类嵌入
    "mid_block_only_cross_attention": None,  # 中间块不使用交叉注意力
    "cross_attention_norm": None,  # 不使用交叉注意力归一化
    "addition_embed_type_num_heads": 64,  # 额外嵌入类型的头数
}

    
    model = UNet2DConditionModel(**unet_config)
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
    #设置编码器
    # 加载预训练的 CLIP 模型
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    #开始训练
    num_epochs = 1000
    best_psnr=float('-inf')
    for epoch in range(num_epochs):
        # validate(model,val_dataloader,scheduler,device,num_inference_steps,epoch)
        train(model,train_dataloader,optimizer,device,epoch,scheduler,tokenizer,text_encoder)
        if (epoch+1) % 10 == 0:
            torch.save(model, "best_model_diffusion.pth")
            psnr=validate(model,val_dataloader,scheduler,device,num_inference_steps,epoch,tokenizer,text_encoder)
            if psnr>best_psnr:
                best_psnr=psnr
                #保存整个模型
                torch.save(model, "best_model.pth")
    
if __name__ == '__main__':
    num_train_timesteps=2000
    num_inference_steps=100
    main(num_train_timesteps,num_inference_steps)