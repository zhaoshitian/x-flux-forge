import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def revise_image(image_path):
    # Open the image file
    image = Image.open(image_path).convert("RGB")
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # print(image_array)
    # print(f"shape of image_array: {image_array.shape}")
    # print("#####################################")
    
    # Invert the colors
    inverted_image_array = 255 - image_array
    # print(inverted_image_array)
    
    # Convert the NumPy array back to an image
    inverted_image = Image.fromarray(inverted_image_array.astype(np.uint8)).convert("L")
    return inverted_image

def resize_and_pad(image_path, desired_width, desired_height):
    # 确保目标宽度和高度是偶数
    desired_width = (desired_width + 1) // 2 * 2
    desired_height = (desired_height + 1) // 2 * 2
    
    # 加载图像
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # 将PIL图像转换为Tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image).unsqueeze(0)  # 添加批次维度
    
    # 获取原始图像尺寸
    original_width, original_height = image.size
    
    # 计算宽高比
    aspect_ratio = original_width / original_height
    
    # 确定新尺寸
    if aspect_ratio < (desired_width / desired_height):
        new_width = int(desired_height * aspect_ratio)
        new_height = desired_height
    else:
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)
    
    # 调整图像大小
    resized_image = F.resize(image_tensor, (new_height, new_width))
    
    # 创建一个新的图像张量，用于填充
    new_image = torch.zeros((3, desired_height, desired_width))
    
    # 计算填充位置
    top = (desired_height - new_height) // 2
    left = (desired_width - new_width) // 2
    
    # 填充新图像张量
    new_image[:, top:top+new_height, left:left+new_width] = resized_image[0]
    
    # 将Tensor转换回PIL图像以查看结果
    transform_to_pil = transforms.ToPILImage()
    padded_image = transform_to_pil(new_image)
    
    return padded_image

class CustomImageDataset(Dataset):
    def __init__(self, data_file_path, desired_width, desired_height):
        self.data_list = json.load(open(data_file_path, "r"))
        # self.img_size = img_size
        self.desired_width = desired_width
        self.desired_height = desired_height

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.data_list[idx]['image'])
            img = resize_and_pad(self.data_list[idx]['image'], self.desired_width, self.desired_height)
            # hint = Image.open(self.data_list[idx]['ocr_result_rendered_image'])
            hint = revise_image(self.data_list[idx]['ocr_result_rendered_image'])
            hint = resize_and_pad(hint, self.desired_width, self.desired_height)
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            # prompt = json.load(open(json_path))['caption']
            prompt = self.data_list[idx]['caption']
            return img, hint, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

if __name__ == "__main__":
    # 使用示例
    image_path = '/data2/stzhao/data/movie_posters-100k/images_png/2.png'  # 替换为你的图像路径
    desired_width = 1000
    desired_height = 1000
    padded_image = resize_and_pad(image_path, desired_width, desired_height)
    padded_image.save("/data2/stzhao/x-flux/image_datasets/test_padding.png")

    render_image_path = "/data2/stzhao/data/movie_posters-100k/render_ocr_result_images/3.png"
    # render_image = Image.open(render_image_path)
    render_image = revise_image(render_image_path)
    render_image.save("/data2/stzhao/x-flux/image_datasets/revise.png")