
import pandas as pd
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, RandomApply, transforms, RandomRotation, RandomVerticalFlip, RandomResizedCrop, RandomHorizontalFlip
from PIL import Image

sampleTransform = Compose([
    # RandomRotation(45),  # 랜덤 각도조절
    # RandomVerticalFlip(),  # 랜덤으로 상하 반전
    # RandomHorizontalFlip(),  # 랜덤으로 좌우 반전
    # #RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.8, 1.25)), # 랜덤 변형
    Resize((224, 224)),
    RandomRotation(45),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    #RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.8, 1.25)),
])


def sampling(ds):
    limit = 300
    # 10000개 이하면 10000개가 될때 까지 데이터 증량

    num_add = limit
    idx = 0

    df = pd.DataFrame()
    while len(df) < limit:
        for i, t in ds.iterrows():
            idx = idx + 1
            img = Image.fromarray(t.waferMap.astype('uint8'))
            img = sampleTransform(img)
            n_img = np.array(img)
            print(n_img.shape)
            df = df.append({
                'waferMap': n_img,
                'encoded_labels': t.encoded_labels,
                'failureType': t.failureType
            }, ignore_index=True)
            if idx == num_add:
                break

    return df
