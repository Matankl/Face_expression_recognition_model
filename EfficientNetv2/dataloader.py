import os
import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FERDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, image_size=224):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_size = image_size
        
        self._load_dataset()
    
    def _load_dataset(self):
        mode_dir = os.path.join(self.root_dir, self.mode)
        class_dirs = os.listdir(mode_dir)
        
        for class_idx, class_name in enumerate(sorted(class_dirs)):
            class_path = os.path.join(mode_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  

        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-15, 15),
            p=0.4
        ),
        A.OneOf([
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7)),
        ], p=0.4),
        A.OneOf([
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ToTensorV2()
        ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ToTensorV2()
    ])

def get_test_transform():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   
        ToTensorV2()
    ])

def test_dataloader(data_dir, batch_size=32, num_workers=4, image_size=224):
    test_dataset = FERDataset(
        root_dir=data_dir,
        mode='PrivateTest',
        transform=get_test_transform(),
        image_size=image_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

def get_train_dataloaders(data_dir, batch_size=32, num_workers=4, image_size=224, data_dir2=None):
    train_dataset = FERDataset(
        root_dir=data_dir,
        mode='train',
        transform=get_train_transform(),
        image_size=image_size
    )
    
    val_dataset = FERDataset(
        root_dir=data_dir,
        mode='validation',
        transform=get_val_transform(),
        image_size=image_size
    )
    if data_dir2 is not None:
        train_dataset2 = FERDataset(
            root_dir=data_dir2,
            mode='Train',
            transform=get_train_transform(),
            image_size=image_size
        )

        val_dataset2 = FERDataset(
            root_dir=data_dir2,
            mode='Test',
            transform=get_val_transform(),
            image_size=image_size
        )
        train_dataset = ConcatDataset([train_dataset, train_dataset2])
        val_dataset = ConcatDataset([val_dataset, val_dataset2])


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader
