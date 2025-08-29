from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import re
import torch

class CustomDataset(VisionDataset):
    def __init__(self, image_folder, mask_folder, image_processor, image_size, subset, num_labels, val_fraction=0.1):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.val_fraction = val_fraction
        self.image_processor = image_processor
        self.num_labels = num_labels
        
        # Define data transformations using Albumentations
        if subset == 'Train': 
            self.transform_base = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.Affine(scale=(0.9, 1.1), translate_percent=0.0625, rotate=(-15, 15), p=0.5),
                            A.OneOf([
                                A.RandomBrightnessContrast(p=1),
                                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
                            ], p=0.3),
                            A.OneOf([
                                A.RandomRain(p=1),
                                A.RandomSunFlare(flare_roi=(0.5, 0.6, 0.7, 0.7), src_radius=350, p=1),
                                A.RandomSnow(p=1),
                                A.RandomFog(alpha_coef=0.3, p=1),
                            ], p=0.3),
                            A.OneOf([
                                A.GridDropout(p=1),
                                A.GaussianBlur(blur_limit=(11, 21), p=1),
                                A.GaussNoise(p=1),
                                A.ISONoise(color_shift=(0.1, 0.5), intensity=(0.6, 0.9), p=1),
                            ], p=0.5),
                            A.RandomResizedCrop(height=image_size[0], width=image_size[1], scale=(0.8, 1.0)),
                            ])
        elif subset == 'Valid':
            self.transform_base = A.Compose([
                            A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_NEAREST),
                            ])
        
        self.transform_img = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                            ToTensorV2(p=1.0),
                            ])
        self.transform_mask = A.Compose([
                            ToTensorV2(p=1.0),
                            ])
        # all files
        self.image_list = np.array(sorted(Path(self.image_folder).glob("*")))
        self.mask_list = np.array(sorted(Path(self.mask_folder).glob("*")))

        for file_path in self.image_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()
        for file_path in self.mask_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()

        self.mask_list = np.array(sorted(self.mask_list, key=lambda path: int(re.findall(r'\d+', path.stem)[0]) if re.findall(r'\d+', path.stem) else 0))

        if subset == 'Train':
            self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.val_fraction)))]
            self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.val_fraction)))]
        elif subset == 'Valid':
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.val_fraction))):]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.val_fraction))):]
        else:
            print('Invalid data subset.')

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        mask_path = self.mask_names[idx]

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:

            image_init = cv2.imread(image_file.name)
            mask_init = cv2.imread(mask_file.name, cv2.IMREAD_GRAYSCALE)
            
            transformed = self.transform_base(image=image_init, mask=mask_init)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            
            ignore = False
            if ignore:
                # This block is currently not used
                pass
            else:
                transformed_mask[transformed_mask==255] = 255
            
            # Using image_processor on image only
            encoded_inputs = self.image_processor(transformed_image, return_tensors="pt")
            image = encoded_inputs["pixel_values"].squeeze(0) # Remove batch dimension

            # Convert mask to tensor separately
            mask = torch.from_numpy(transformed_mask).long()
            
            return [image, mask]