import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import ast
from typing import Callable, Optional, Tuple

class ODIRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading ODIR-5K fundus images and labels.
    """
    def __init__(self, csv_path: str, img_dir: str, transform: Optional[Callable] = None):
        """
        Args:
            csv_path (str): Path to the partition CSV file (train, val, or test).
            img_dir (str): Directory containing preprocessed fundus images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load image
        img_name = row['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Open image and ensure RGB mode
        image = Image.open(img_path).convert('RGB')
        
        # Parse target string to float list, then to Tensor
        # Ex: "[1, 0, 0, 0, 0, 0, 0, 0]" -> [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        target_list = ast.literal_eval(row['target'])
        target = torch.tensor(target_list, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
            
        return image, target
