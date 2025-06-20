import os
import zipfile
import urllib.request
from typing import Optional, Callable, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Cotton80Dataset(Dataset):
    """
    Ultra-FGVC Cotton 80 Dataset for PyTorch
    
    Args:
        root (str): Root directory where dataset will be stored
        split (str): Dataset split - 'train', 'val', or 'test'
        transform (callable, optional): Optional transform to be applied on samples
        target_transform (callable, optional): Optional transform to be applied on targets
        download (bool, optional): If True, downloads the dataset if it doesn't exist
        zip_url (str, optional): URL to download the dataset zip file
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        zip_url: str = "https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true"
    ):
        self.root = root
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.zip_url = zip_url
        
        # Validate split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        
        # Create root directory if it doesn't exist
        os.makedirs(root, exist_ok=True)
        
        # Download dataset if requested
        if download:
            self._download()
        
        # Check if dataset exists
        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it'
            )
        
        # Load annotations
        self.samples = self._load_annotations()
        
        # Get unique classes and create class mapping
        self.classes = sorted(list(set(label for _, label in self.samples)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
    
    def _check_exists(self) -> bool:
        """Check if the dataset files exist"""
        cotton_dir = os.path.join(self.root, 'COTTON')
        anno_dir = os.path.join(cotton_dir, 'anno')
        images_dir = os.path.join(cotton_dir, 'images')
        anno_file = os.path.join(anno_dir, f'{self.split}.txt')
        
        return (os.path.exists(cotton_dir) and 
                os.path.exists(anno_dir) and 
                os.path.exists(images_dir) and 
                os.path.exists(anno_file))
    
    def _download(self):
        """Download and extract the dataset"""
        if self._check_exists():
            print("Dataset already exists. Skipping download.")
            return
        
        if self.zip_url == "<place_holder>":
            raise ValueError("Please provide a valid zip_url for downloading the dataset")
        
        zip_path = os.path.join(self.root, 'Cotton80.zip')
        
        print(f"Downloading Cotton80 dataset from {self.zip_url}")
        try:
            urllib.request.urlretrieve(self.zip_url, zip_path)
            print("Download completed.")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root)
            print("Extraction completed.")
            
            # Clean up zip file
            os.remove(zip_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")
    
    def _load_annotations(self) -> List[Tuple[str, int]]:
        """Load annotations from the split file"""
        cotton_dir = os.path.join(self.root, 'COTTON')
        anno_file = os.path.join(cotton_dir, 'anno', f'{self.split}.txt')
        
        samples = []
        with open(anno_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        label = int(parts[1]) - 1
                        samples.append((image_name, label))
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset"""
        if idx >= len(self.samples):
            raise IndexError("Index out of range")
        
        image_name, label = self.samples[idx]
        
        # Load image
        cotton_dir = os.path.join(self.root, 'COTTON')
        image_path = os.path.join(cotton_dir, 'images', image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the current split"""
        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Split: {self.split}")
        body.append(f"Root location: {self.root}")
        body.append(f"Number of classes: {self.num_classes}")
        if hasattr(self, 'transform') and self.transform is not None:
            body.append(f"Transforms: {self.transform}")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body.append(f"Target transforms: {self.target_transform}")
        
        lines = [head] + ["    " + line for line in body]
        return '\n'.join(lines)


# Example usage and utility functions
def get_default_transforms(split: str = 'train', image_size: int = 224):
    """Get default transforms for the dataset"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(root: str, batch_size: int = 32, num_workers: int = 4, 
                      download: bool = False, zip_url: str = "<place_holder>", transform=None):
    """Create DataLoaders for all splits"""
    from torch.utils.data import DataLoader
    image_transform = transform if transform is not None else get_default_transforms('train')
    # Create datasets
    train_dataset = Cotton80Dataset(
        root=root, 
        split='train', 
        transform=image_transform,
        download=download,
        zip_url=zip_url
    )
    
    val_dataset = Cotton80Dataset(
        root=root, 
        split='val', 
        transform=image_transform,
        download=False,  # Already downloaded with train
        zip_url=zip_url
    )
    
    test_dataset = Cotton80Dataset(
        root=root, 
        split='test', 
        transform=image_transform,
        download=False,  # Already downloaded with train
        zip_url=zip_url
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    dataset = Cotton80Dataset(
        root='./data',
        split='train',
        transform=get_default_transforms('train'),
        download=True,  # Set to True to download
        zip_url="https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true"  # Replace with actual URL
    )
    
    print(dataset)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Get a sample
    image, label = dataset[0]
    print(f"Sample shape: {image.shape}, Label: {label}")
    
    # Example 2: Create all dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        root='./data',
        batch_size=32,
        download=True,
        zip_url="https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true"  # Replace with actual URL
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Total classes: {num_classes}")
    
    # Example 3: Iterate through a batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        if batch_idx == 0:  # Just show first batch
            break