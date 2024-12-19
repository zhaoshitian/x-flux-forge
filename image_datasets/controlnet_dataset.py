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

def resize(
    image: np.ndarray,
    size: Tuple[int, int],
    resample: "PILImageResampling" = None,
    reducing_gap: Optional[int] = None,
    data_format: Optional[ChannelDimension] = None,
    return_numpy: bool = True,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Resizes `image` to `(height, width)` specified by `size` using the PIL library.

    Args:
        image (`np.ndarray`):
            The image to resize.
        size (`Tuple[int, int]`):
            The size to use for resizing the image.
        resample (`int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            The filter to user for resampling.
        reducing_gap (`int`, *optional*):
            Apply optimization by resizing the image in two steps. The bigger `reducing_gap`, the closer the result to
            the fair resampling. See corresponding Pillow documentation for more details.
        data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the output image. If unset, will use the inferred format from the input.
        return_numpy (`bool`, *optional*, defaults to `True`):
            Whether or not to return the resized image as a numpy array. If False a `PIL.Image.Image` object is
            returned.
        input_data_format (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If unset, will use the inferred format from the input.

    Returns:
        `np.ndarray`: The resized image.
    """
    requires_backends(resize, ["vision"])

    resample = resample if resample is not None else PILImageResampling.BILINEAR

    if not len(size) == 2:
        raise ValueError("size must have 2 elements")

    # For all transformations, we want to keep the same data format as the input image unless otherwise specified.
    # The resized image from PIL will always have channels last, so find the input format first.
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # To maintain backwards compatibility with the resizing done in previous image feature extractors, we use
    # the pillow library to resize the image and then convert back to numpy
    do_rescale = False
    if not isinstance(image, PIL.Image.Image):
        do_rescale = _rescale_for_pil_conversion(image)
        image = to_pil_image(image, do_rescale=do_rescale, input_data_format=input_data_format)
    height, width = size
    # PIL images are in the format (width, height)
    resized_image = image.resize((width, height), resample=resample, reducing_gap=reducing_gap)

    if return_numpy:
        resized_image = np.array(resized_image)
        # If the input image channel dimension was of size 1, then it is dropped when converting to a PIL image
        # so we need to add it back if necessary.
        resized_image = np.expand_dims(resized_image, axis=-1) if resized_image.ndim == 2 else resized_image
        # The image is always in channels last format after converting from a PIL image
        resized_image = to_channel_dimension_format(
            resized_image, data_format, input_channel_dim=ChannelDimension.LAST
        )
        # If an image was rescaled to be in the range [0, 255] before converting to a PIL image, then we need to
        # rescale it back to the original range.
        resized_image = rescale(resized_image, 1 / 255) if do_rescale else resized_image
    return resized_image


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


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

def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    if channel_dim == ChannelDimension.FIRST:
        return image.shape[-2], image.shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image.shape[-3], image.shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")

def infer_channel_dimension_format(
    image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    if image.shape[first_dim] in num_channels and image.shape[last_dim] in num_channels:
        logger.warning(
            f"The channel dimension is ambiguous. Got image shape {image.shape}. Assuming channels are the first dimension."
        )
        return ChannelDimension.FIRST
    elif image.shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image.shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")

def custom_collate_fn_dynamres(batch):

    min_pixels = 56 * 56
    max_pixels = 14 * 14 * 4 * 1280

    images = []
    hints = []
    prompts = []
    for item in batch:
        img, hint, prompt = item
        images.append(img)
        hints.append(hint)
        prompts.append(prompt)
    
    # tensors = [item for item in batch]

        input_data_format = infer_channel_dimension_format(img)

        height, width = get_image_size(img, channel_dim=input_data_format)
        resized_height, resized_width = height, width

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        image = resize(
            img, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
        )

    return torch.stack(tensors, dim=0)


def custom_collate_fn_staticres(batch):

    try:
        img = Image.open(self.data_list[idx]['image'])
        img = resize_and_pad(self.data_list[idx]['image'], self.desired_width, self.desired_height)
        # hint = Image.open(self.data_list[idx]['ocr_result_rendered_image'])
        hint = revise_image(self.data_list[idx]['ocr_result_rendered_image'])
        hint = resize_and_pad(hint, self.desired_width, self.desired_height)
        img = torch.from_numpy((np.array(img) / 127.5) - 1)
        img = img.permute(2, 0, 1)
        # print(f"shape of image: {img.shape}")
        hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
        hint = hint.permute(2, 0, 1)
        # print(f"shape of hint: {hint.shape}")
        # prompt = json.load(open(json_path))['caption']
        prompt = self.data_list[idx]['caption']
        return img, hint, prompt
    except Exception as e:
        print(e)
        return self.__getitem__(random.randint(0, len(self.data_list) - 1))

    return torch.stack(tensors, dim=0)



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
            # print(f"shape of image: {img.shape}")
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            # print(f"shape of hint: {hint.shape}")
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