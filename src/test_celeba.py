import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# 使用したい属性名
used_attrs = ['Smiling', 'Male', 'Young']
# used_attrs = ['5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive']
attr_indices = None  # 後で取得する

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # [0, 1]
])

# CelebA データセットの読み込み（初回はDLに時間がかかる）
celeba = datasets.CelebA(
    root='./data', 
    split='train', 
    target_type='attr',
    transform=transform, 
    download=False
)

# 属性名 → インデックスの対応表
if attr_indices is None:
    attr_names = celeba.attr_names
    # print(attr_names)
    # print(len(attr_names))
    attr_indices = [attr_names.index(attr) for attr in used_attrs]

# 属性付きデータセットとしてラップ
class CelebAWithSelectedAttrs(torch.utils.data.Dataset):
    def __init__(self, celeba, attr_indices):
        self.celeba = celeba
        self.attr_indices = attr_indices

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        img, attrs = self.celeba[idx]
        selected_attrs = attrs[self.attr_indices].float()  # shape: (3,)
        return img, selected_attrs

dataset = CelebAWithSelectedAttrs(celeba, attr_indices)
dataloader_wrapper = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader = DataLoader(celeba, batch_size=1, shuffle=False)

print(next(iter(dataloader)))
print(next(iter(dataloader_wrapper)))
