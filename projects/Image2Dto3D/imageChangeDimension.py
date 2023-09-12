
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import pandas as pd
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, RandomApply, transforms, RandomRotation, RandomVerticalFlip, RandomResizedCrop, RandomHorizontalFlip
from PIL import Image

# 2차원 -> 3차원 변환


def tran3d(ds, limit):
    count = 0
    for idx, t in ds.iterrows():
        if count == limit:
            break
        count = count+1
        img = Image.fromarray(t.waferMap.astype('uint8'))
        # resized_img = img.resize((224, 224))
        arr_resized = np.asarray(img)

        image_shape = np.shape(arr_resized)
        arr_3d = np.zeros(
            (image_shape[0], image_shape[1], 3), dtype=np.float32)

        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                if arr_resized[i][j] == 0:
                    arr_3d[i][j] = [0, 0, 0]  # 검정
                elif arr_resized[i][j] == 1:
                    arr_3d[i][j] = [0, 200, 255]  # 청록
                elif arr_resized[i][j] == 2:
                    arr_3d[i][j] = [200, 50, 0]  # 노랑

        img = torch.tensor(arr_3d, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        ds.at[idx, 'waferMap'] = img

    return ds[:limit]
