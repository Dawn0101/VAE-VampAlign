# data_loaders.py 负责加载两个训练数据集

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, torch

# 在 data_loaders.py 开头加：
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomCelebADataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(root, 'img_align_celeba')
        
        # 读取划分文件
        with open(os.path.join(root, 'list_eval_partition.txt'), 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
        # 分割映射: 0=train, 1=val, 2=test
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        self.filenames = [fname for fname, part in lines if int(part) == split_id]
        
        # 读取属性（可选）
        with open(os.path.join(root, 'list_attr_celeba.txt'), 'r') as f:
            attr_lines = f.readlines()
        self.attr_names = attr_lines[1].strip().split()
        self.attr_dict = {}
        for line in attr_lines[2:]:
            parts = line.strip().split()
            self.attr_dict[parts[0]] = torch.tensor([int(x) for x in parts[1:]], dtype=torch.float32)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.attr_dict[fname]  # [40] tensor of -1/1
        return img, label
    
class MNISTDataModule:
    def __init__(self, config):
        self.config = config
        self.data_path = config['data']['mnist_data_path']
        self.img_size = config['train']['image_size']
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)

        os.makedirs(self.data_path, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def get_dataset(self, train: bool = True):
        return datasets.MNIST(
            root=self.data_path,
            train=train,
            download=True,
            transform=self.transform
        )

    def get_loader(self, train: bool = True):
        dataset = self.get_dataset(train=train)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True
        )

class CelebADataModule:
    REQUIRED_FILES = [
        "list_attr_celeba.txt",
        "list_eval_partition.txt",
        "img_align_celeba"
    ]

    def __init__(self, config):
        self.config = config
        self.data_path = config['data']['celebA_data_path']
        self.img_size = config['train']['image_size']
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)

        self._validate_celeba_structure()

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])

    def _validate_celeba_structure(self):
        missing = []
        for f in self.REQUIRED_FILES:
            full_path = os.path.join(self.data_path, f)
            if not os.path.exists(full_path):
                missing.append(f)
        if missing:
            raise FileNotFoundError(
                f"CelebA dataset incomplete at {self.data_path}. Missing: {missing}\n"
                "Please download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n"
                "Required: img_align_celeba/ (dir), list_attr_celeba.txt, list_eval_partition.txt"
            )

    def get_dataset(self, split: str = 'train'):
        return CustomCelebADataset(
            root=self.data_path,
            split=split,
            transform=self.transform
        )

    def get_loader(self, split: str = 'train'):
        dataset = self.get_dataset(split=split)
        shuffle = (split == 'train')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
