#!/usr/bin/env python3
"""
将NIfTI格式的MRI数据转换为PNG格式的2D切片
- 提取矢状面（sagittal）切片
- 只保存包含标注的切片（mask中有非0值）
- 输出PNG格式图像
"""

import argparse
import logging
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def normalize_image(img_array):
    """将图像归一化到0-255范围"""
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max > img_min:
        normalized = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img_array, dtype=np.uint8)
    return normalized


def extract_sagittal_slices(nii_path, mask_path, output_img_dir, output_mask_dir, base_name):
    """
    从NIfTI文件中提取矢状面切片
    
    Args:
        nii_path: 原始MRI图像路径
        mask_path: 标注mask路径
        output_img_dir: 输出图像目录
        output_mask_dir: 输出mask目录
        base_name: 基础文件名（不含扩展名）
    """
    # 加载NIfTI文件
    img_nii = nib.load(nii_path)
    mask_nii = nib.load(mask_path)
    
    # 获取数据数组
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata().astype(np.uint8)
    
    # 确保图像和mask尺寸一致
    assert img_data.shape == mask_data.shape, \
        f"Image shape {img_data.shape} != mask shape {mask_data.shape}"
    
    # 矢状面切片：沿着第三个维度（最后一个维度）
    # 对于shape为(H, W, D)的NIfTI数据，矢状面是沿着第三个维度提取
    num_slices = img_data.shape[2]
    
    saved_count = 0
    
    for slice_idx in range(num_slices):
        # 提取矢状面切片：data[:, :, slice_idx]
        img_slice = img_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx]
        
        # 只保存包含标注的切片（mask中有非0值）
        if np.any(mask_slice > 0):
            # 归一化图像
            img_normalized = normalize_image(img_slice)
            
            # 确保mask值为0和1（将任何非0值转换为1）
            # 注意：实际数据中可能使用2或其他值表示目标，统一转换为1
            mask_binary = (mask_slice > 0).astype(np.uint8) * 255
            
            # 创建文件名：base_name_slice_XXX.png
            slice_name = f"{base_name}_slice_{slice_idx:03d}.png"
            
            # 保存图像
            img_pil = Image.fromarray(img_normalized, mode='L')  # L模式表示灰度图
            img_pil.save(output_img_dir / slice_name)
            
            # 保存mask
            mask_pil = Image.fromarray(mask_binary, mode='L')
            mask_pil.save(output_mask_dir / slice_name)
            
            saved_count += 1
    
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='将NIfTI格式转换为PNG格式的2D切片')
    parser.add_argument('--input-img-dir', type=str, default='original_nii_data/0-un-label',
                        help='输入原始MRI图像目录')
    parser.add_argument('--input-mask-dir', type=str, default='original_nii_data/1-labelled',
                        help='输入标注mask目录')
    parser.add_argument('--output-img-dir', type=str, default='data/imgs',
                        help='输出图像目录')
    parser.add_argument('--output-mask-dir', type=str, default='data/masks',
                        help='输出mask目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_img_dir = Path(args.output_img_dir)
    output_mask_dir = Path(args.output_mask_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 输入目录
    input_img_dir = Path(args.input_img_dir)
    input_mask_dir = Path(args.input_mask_dir)
    
    # 获取所有NIfTI文件
    img_files = sorted(input_img_dir.glob('*.nii.gz'))
    
    if not img_files:
        logging.error(f'在 {input_img_dir} 中未找到NIfTI文件')
        return
    
    logging.info(f'找到 {len(img_files)} 个NIfTI文件')
    
    total_slices = 0
    
    # 处理每个文件
    for img_file in tqdm(img_files, desc='处理文件'):
        # 获取基础文件名（不含扩展名）
        base_name = img_file.stem.replace('.nii', '')
        
        # 查找对应的mask文件
        # 尝试多种可能的mask文件名
        mask_file = input_mask_dir / img_file.name
        
        if not mask_file.exists():
            # 尝试其他可能的命名方式
            # 例如：MR0524767-Right-sag-label.nii.gz 对应 MR0524767-sag.nii.gz
            possible_names = [
                img_file.name,
                img_file.name.replace('-sag.nii.gz', '-sag-label.nii.gz'),
                img_file.name.replace('.nii.gz', '-label.nii.gz'),
            ]
            
            mask_file = None
            for name in possible_names:
                candidate = input_mask_dir / name
                if candidate.exists():
                    mask_file = candidate
                    break
            
            if mask_file is None:
                logging.warning(f'未找到 {img_file.name} 对应的mask文件，跳过')
                continue
        
        try:
            saved = extract_sagittal_slices(
                img_file, mask_file, output_img_dir, output_mask_dir, base_name
            )
            total_slices += saved
            logging.info(f'{base_name}: 保存了 {saved} 个切片')
        except Exception as e:
            logging.error(f'处理 {img_file.name} 时出错: {e}')
            continue
    
    logging.info(f'转换完成！总共保存了 {total_slices} 个切片')
    logging.info(f'图像保存在: {output_img_dir}')
    logging.info(f'Mask保存在: {output_mask_dir}')


if __name__ == '__main__':
    main()

