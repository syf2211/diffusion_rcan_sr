import os
import tifffile
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
# from basicsr.utils.registry import DATASET_REGISTRY
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transform
import random
from tifffile import imwrite
# from basicsr.data.transforms import paired_random_crop
def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs
class MyDataset(Dataset):
    def __init__(self, 
                opt, 
                is_train=True, 
                autosnr=True, 
                use_num=False,
                num_top_pixels=1000,
                aug_prob=0.0):
        # print("start\n")
        # print(opt)
        self.opt = opt
        self.is_train = is_train
        self.autosnr = autosnr
        self.use_num = use_num
        self.num_top_pixels = num_top_pixels
        self.aug_prob = aug_prob
        # self.overexposure = opt['overexposure_method']
        self.struc = opt['structure']
        self.struc_sel = opt['structure_selection']
        self.snr = opt['snr']
        self.N = None

        self.name_list = []
        self.filename = {}
        self.filenum = {}
        self.threholds = []

        self.success=0#debug
  
        if not self.struc_sel:
            print("no selection")
            if self.is_train:
                self.path = f"{opt['train_dataroot']}/"
                text_path = f"{opt['train_dataroot']}/train.json"
                with open(text_path, 'r') as f:
                    self.text = json.load(f)
            else:
                self.path = f"{opt['val_dataroot']}/"
                text_path = f"{opt['val_dataroot']}/val.json"
                with open(text_path, 'r') as f:
                    self.text = json.load(f)
            self.name_list=self.struc  #syf
        else:
            if self.is_train:
                struc_num = len(self.struc)
                self.text = self.struc
                self.path = f"{opt['train_dataroot']}/"
                # skip_else=False
                # threhold mat not finished, it should accompanied by a adapted dataset folder structure, but haven't changed yet(in data_agmt_matlab)
                for i in range(struc_num):
                    threhold = []
                    t = 0
                    if self.struc_sel[i]==1:
                        # skip_else=True
                        self.name_list.append(self.struc[i])#syf
                        self.filename[self.struc[i]] = {}
                        self.filenum[self.struc[i]] = {}
                        for j in range(len(self.snr)):
                            self.filename[self.struc[i]][self.snr[j]] = []
                            for item in os.listdir(f"{opt['train_dataroot']}/{self.struc[i]}/gt/{self.snr[j]}"):
                                self.filename[self.struc[i]][self.snr[j]].append(item)
                            t += len(self.filename[self.struc[i]][self.snr[j]])
                            threhold.append(t)
                            self.filenum[self.struc[i]][self.snr[j]] = len(self.filename[self.struc[i]][self.snr[j]])
                        self.threholds.append(threhold)
                    
                        
                self.threholds = np.array(self.threholds)
            else:
                struc_num = len(self.struc)
                self.text = self.struc
                self.path = f"{opt['val_dataroot']}/"
                # threhold mat not finished, it should accompanied by a adapted dataset folder structure, but haven't changed yet(in data_agmt_matlab)
                for i in range(struc_num):
                    threhold = []
                    t = 0
                    if self.struc_sel[i]==1:
                        self.name_list.append(self.struc[i])#syf
                        self.filename[self.struc[i]] = {}
                        self.filenum[self.struc[i]] = {}
                        for j in range(len(self.snr)):
                            self.filename[self.struc[i]][self.snr[j]] = []
                            for item in os.listdir(f"{opt['val_dataroot']}/{self.struc[i]}/gt/{self.snr[j]}"):
                                self.filename[self.struc[i]][self.snr[j]].append(item)
                            t += len(self.filename[self.struc[i]][self.snr[j]])
                            threhold.append(t)
                            self.filenum[self.struc[i]][self.snr[j]] = len(self.filename[self.struc[i]][self.snr[j]])
                        self.threholds.append(threhold)

                self.threholds = np.array(self.threholds)

        print("start\n")
        print(self.threholds)
        # breakpoint()
    def __len__(self):
        if not self.struc_sel:
            file_count = 0
            if not self.N:
                for item in os.listdir(self.lq_path):
                    if os.path.isfile(os.path.join(self.lq_path, item)):
                        file_count += 1
            else:
                file_count = self.N
        else:
            if not self.N:
                file_count = np.sum(self.threholds[:, -1])
                self.N = file_count
            else:
                file_count = self.N

        return file_count

    def get_transforms(self):
        transform_list = []
        # if np.random.randn(1) < self.opt['aug_prob']:
            # transform_list.append(transforms.CenterCrop(256))
        if np.random.randn(1) < self.aug_prob:
            transform_list.append(transforms.RandomRotation(15))
        if np.random.randn(1) < self.aug_prob:
            transform_list.append(transforms.RandomHorizontalFlip())

        return transform_list

    def idx2path(self, idx):
        num = len(self.name_list)
        img_name = f'{(idx + 1):04d}.tif'

        for i in range(num):
            if idx < np.sum(self.threholds[0:i+1, -1]):
                text = self.name_list[i]
                if i > 0:
                    idx = idx - np.sum(self.threholds[0:i, -1])  # idx donot need syf
                for j in range(len(self.snr)):
                    if idx < np.sum(self.threholds[i][j]):#j+1 previous
                        if j > 0:
                            idx = idx - np.sum(self.threholds[i][j-1])
                        img_name = f'{(idx + 1):04d}.tif'
                        gt_path = f'{self.path}/{text}/gt/{self.snr[j]}/'
                        lq_path = f'{self.path}/{text}/wf/{self.snr[j]}/'
                        break
                break
        
        return gt_path, lq_path, img_name, text
    
    def load_images(self, gt_path, lq_path, image_name, text):
        gt_path = f'{gt_path}/{image_name}'
        lq_path = f'{lq_path}/{image_name}'
        gt_temp=tifffile.imread(gt_path)
        # gt = tifffile.imread(gt_path).astype(np.float32)
        # lq = tifffile.imread(lq_path).astype(np.float32)
        gt =tifffile.imread(gt_path)
        lq =tifffile.imread(lq_path)
        # print(np.max(gt)-np.min(gt))
        # print(np.max(gt),np.min(gt))
        # lq = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        #存储gt
        
        # if not self.is_train:
        #     gt_before=gt
        #     gt_before=gt_before
        #     imwrite(f"gt_{image_name}", gt_before)

        # random crop
        if self.is_train:
            gt_size = self.opt['gt_size']
            gt, lq = paired_random_crop(gt, lq, gt_size, 1)
            # safe_file_name = gt_path.replace('/', '_').replace(':', '_').replace(' ', '_') 
            # tifffile.imsave(f'./temp_image/{safe_file_name}_{image_name}', gt)
            # tifffile.imsave(f'./temp_image/lq_{image_name}', lq)
        gt_min = np.min(gt)
        gt_max = np.max(gt)
        lq_min = np.min(lq)
        lq_max = np.max(lq)
        
        # add snr 
        if self.autosnr:
            flattened_lq = lq.flatten()
        
            num_top_pixels = self.num_top_pixels
            top_pixels = np.partition(flattened_lq, -num_top_pixels)[-num_top_pixels:]
            mean_top_pixels = np.mean(top_pixels)

            if self.use_num:
                if self.struc_sel.count(1)==1:
                    text=''
                else:
                    text = f'{text}, snr {int(mean_top_pixels)}'
            else:
                if self.struc_sel.count(1)==1:
                    text=''
                else:
                    text=f'{text}'
        # if self.is_train:
        #     gt = (gt - gt_min) / (gt_max - gt_min)
        #     lq = (lq - lq_min) / (lq_max - lq_min)
        #标准化
        if self.is_train:
            gt= (gt-gt.mean())/gt.std()
            lq= (lq-lq.mean())/lq.std()
        gt = torch.unsqueeze(torch.from_numpy(gt), 0)
        lq = torch.unsqueeze(torch.from_numpy(lq), 0)
        # if not self.is_train:
        #     #存储gt
        #     gt_before=gt.squeeze(0).cpu().numpy()
        #     gt_before=gt_before
        #     imwrite(f"gt_before_.tiff", gt_before)

        return gt, lq, text

    def __getitem__(self, idx):
        if not self.struc_sel:
            img_name = f'{(idx + 1):04d}.tif'
            text = self.text[img_name]
            gt_path = f'{self.path}/gt/'
            lq_path = f'{self.path}/lq/'
        else:
            gt_path, lq_path, img_name, text = self.idx2path(idx)
        
        gt, lq, text = self.load_images(gt_path, lq_path, img_name, text)
        
        if self.is_train:
            train = {'gt': gt, 'lq': lq, 'text': text}
            # breakpoint()
            self.success+=1
            return train
        else:
            val = {'gt': gt, 'lq': lq, 'text': text, 'lq_path': f'{lq_path}/{img_name}'}
            return val



