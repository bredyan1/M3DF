# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import pdb

import PIL
import SimpleITK as sitk
import numpy as np
import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from torchvision import datasets, transforms

from timm.data import create_transform  # 根据指定的参数配置创建一个图像数据预处理的转换器可以包括对图像进行大小调整、裁剪、标准化、数据增强等操作
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

output_size = (128, 128, 128)

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    # both_none = extensions is None and is_valid_file is None
    # both_something = extensions is not None and is_valid_file is not None
    # if both_none or both_something:
    #     raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, fnames, _ in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if os.path.isdir(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        # if extensions is not None:
        #     msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        # raise FileNotFoundError(msg)
    # pdb.set_trace()
    return instances
class DICOMFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.loader = loader


    # def _find_classes(self, dir):
    #     classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     classes.sort()
    #     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    #     return classes, class_to_idx
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cant be None.")
        return make_dataset(directory, class_to_idx, extensions=None, is_valid_file=is_valid_file)


names=[]
def my_loader(path: str):
    try:
        # 读取输入文件夹中的所有 DICOM 文件
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        #print(dicom_names)
        if not dicom_names:
            raise RuntimeError(f"在目录 {path} 中未找到 DICOM 文件")

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        name = os.path.basename(os.path.dirname(dicom_names[0]))
        names.append(name)

        # 设置输出图像的大小
        new_spacing = [(old_sz * old_spc) / new_sz for old_sz, old_spc, new_sz in
                       zip(image.GetSize(), image.GetSpacing(), output_size)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(output_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)

        # 执行重采样
        output_image = resampler.Execute(image)

        # 将 SimpleITK 图像转换为 NumPy 数组，并将像素值转换为浮点数类型
        output_array = sitk.GetArrayFromImage(output_image).astype(np.float32)

        # 归一化操作
        normalized_array = output_array / 1000.0  # 这个除以一千的操作不太严谨，你自己再斟酌一下

        # 将 NumPy 数组转换为 PyTorch Tensor
        output_tensor = torch.tensor(normalized_array, dtype=torch.float32)
        output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)
        
        print(output_tensor.shape)
        return output_tensor

    except Exception as e:
        print(f"加载图像时出错：{e}")
        return None
# def my_loader(path: str):
#     # 读取输入文件夹中的所有 DICOM 文件
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(path)
#     reader.SetFileNames(dicom_names)
#     image = reader.Execute()
#     global names
#     name=os.path.basename(os.path.dirname(dicom_names[0]))
#     names.append(name)
#     # 设置输出图像的大小
#     new_spacing = [(old_sz * old_spc) / new_sz for old_sz, old_spc, new_sz in
#                    zip(image.GetSize(), image.GetSpacing(), output_size)]
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetSize(output_size)
#     resampler.SetOutputSpacing(new_spacing)
#     resampler.SetOutputDirection(image.GetDirection())
#     resampler.SetOutputOrigin(image.GetOrigin())
#     resampler.SetInterpolator(sitk.sitkLinear)

#     # 执行重采样
#     output_image = resampler.Execute(image)

#     # 将 SimpleITK 图像转换为 NumPy 数组，并将像素值转换为浮点数类型
#     output_array = sitk.GetArrayFromImage(output_image).astype(np.float32)

#     normalized_array = output_array / 1000#这个除以一千的操作不太严谨，你自己再斟酌一下

#     # 将 NumPy 数组转换为 PyTorch Tensor
#     output_tensor = torch.tensor(normalized_array, dtype=torch.float32)  # 30*224*224
#     output_tensor = output_tensor.unsqueeze(0)
#     output_tensor = output_tensor.unsqueeze(0)
#     print(output_tensor.shape)
#     return output_tensor


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path)
    dataset= DICOMFolder(root, transform=None, loader = my_loader)
    global names
    print(dataset)

    return dataset,names


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    # t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_transform_mri(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
