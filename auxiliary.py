# Defines the classes and used functions from the data jupyter notebook 
# This avoids clutering and improves readability

from pathlib import Path
import copy
import numbers
import random
import time
import zipfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
from skimage import io, transform, util, exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
SEED = 42
CLASS_NAMES = ['Water', 'Very Low', 'Low', 'Moderate',
               'High', 'Very High', 'Non-burnable']
NUM_CLASSES = len(CLASS_NAMES)
MAX_TRAIN_SAMPLES = 0  # Set to 0 to use the full training split.
NUM_WORKERS = 0
CUSTOM_EPOCHS = 5
FT_EPOCHS = 10
CUSTOM_LR = 1e-2
FT_LR = 1e-3

# Note: PIXEL_MEAN and PIXEL_STD come directly from AlexNet's
# documentation in PyTorch:
# https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

DATA_ARCHIVE = Path('db.zip')
DATA_ROOT = Path('data')
DB_DIR = DATA_ROOT / 'db'
OUTPUT_DIR = Path('submissions')

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

##### AUGMENTATION METHODS
# It allows for multiple augmentation methods
class Compose:
    """Applies a list of dictionary-based transforms in sequence."""
    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

## EXAMPLE CASE:
# train_transform = Compose([
#     SkResize((300, 300)),
#     TVRandomHorizontalFlip(p=0.5),
#     TVRandomRotation(degrees=15),
#     TVColorJitter(brightness=0.1, contrast=0.1),
#     TVCenterCrop(256),
#     SkAddGaussianNoise(var=0.005)
# ])


def _crop_size(output_size):

    if isinstance(output_size, numbers.Integral):
        output_size = (int(output_size), int(output_size))

    if (not isinstance(output_size, tuple) or len(output_size) != 2 or
        any(not isinstance(v, numbers.Integral) or v <= 0 for v in output_size)):
        raise ValueError('Size must be a positive integer or a pair of positive integers.')

    return tuple(int(v) for v in output_size)

class Rescale:
    """Resize to (height, width), or resize the shorter side preserving aspect ratio."""

    def __init__(self, output_size):
        self.output_size = (int(output_size) if isinstance(output_size, numbers.Integral)
                            else _crop_size(output_size))
        if isinstance(self.output_size, int) and self.output_size <= 0:
            raise ValueError('The output size must be positive.')

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            scale = self.output_size / min(h, w)
            new_h, new_w = round(h * scale), round(w * scale)
        else:
            new_h, new_w = self.output_size
        image = transform.resize(image, (new_h, new_w), anti_aliasing=True)

        return {'image': image.astype(np.float32), 'label': label}

class RandomCrop:
    """Choose uniformly among all valid crops, including the final row/column."""

    def __init__(self, output_size):
        self.output_size = _crop_size(output_size)

    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if new_h > h or new_w > w:
            raise ValueError('Crop exceeds image dimensions; resize the image first.')
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        return {'image': image[top:top + new_h, left:left + new_w], 'label': label}

class CenterCrop:
    """Take a deterministic central crop without silently changing its size."""

    def __init__(self, output_size):
        self.output_size = _crop_size(output_size)

    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if new_h > h or new_w > w:
            raise ValueError('Crop exceeds image dimensions; resize the image first.')
        top, left = (h - new_h) // 2, (w - new_w) // 2

        return {'image': image[top:top + new_h, left:left + new_w], 'label': label}

class ToTensor:
    """Convert HWC images to contiguous float32 CHW tensors and labels to int64."""

    def __call__(self, sample):
        image = np.ascontiguousarray(sample['image'].transpose(2, 0, 1), dtype=np.float32)
        return {'image': torch.from_numpy(image),
                'label': torch.tensor(sample['label'], dtype=torch.long)}

class Normalize:
    """Apply per-channel normalisation without mutating the input tensor."""

    def __init__(self, mean, std):

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

        if (self.mean.ndim != 1 or self.std.shape != self.mean.shape or
            not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or
            not (self.std > 0).all()):
            raise ValueError('Mean/std must be matching finite vectors with positive std.')

    def __call__(self, sample):

        image = sample['image']

        if image.ndim != 3 or image.shape[0] != len(self.mean):
            raise ValueError('The CHW tensor and normalisation vectors must match.')

        mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)[:, None, None]
        return {'image': (image - mean) / std, 'label': sample['label']}

# Adapts Pytorch's Center Crop to work on dictionary style dataset samples containing NP array images
class TVCenterCrop:
    """Wrap torchvision CenterCrop for a NumPy image/sample dictionary."""

    def __init__(self, size):
        self.size = _crop_size(size)
        self.crop = transforms.CenterCrop(self.size)

    def __call__(self, sample):

        image = sample['image']

        if any(crop > actual for crop, actual in zip(self.size, image.shape[:2])):
            raise ValueError('Crop exceeds image dimensions; resize the image first.')

        pil_image = Image.fromarray(util.img_as_ubyte(image))
        image = util.img_as_float32(np.asarray(self.crop(pil_image)))
        return {'image': image, 'label': sample['label']}

# Torch vision transformations
class TVRandomHorizontalFlip:
    """Randomly flips the image horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = p
        self.flip = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            pil_img = Image.fromarray(util.img_as_ubyte(image))
            image = util.img_as_float32(np.asarray(self.flip(pil_img)))
            return {'image': image, 'label': sample['label']}
        return sample


class TVRandomRotation:
    """Rotates the image by a random angle within [-degrees, +degrees]."""
    def __init__(self, degrees=15):
        self.rotate = transforms.RandomRotation(degrees=degrees)

    def __call__(self, sample):
        image = sample['image']
        pil_img = Image.fromarray(util.img_as_ubyte(image))
        image = util.img_as_float32(np.asarray(self.rotate(pil_img)))
        return {'image': image, 'label': sample['label']}


class TVColorJitter:
    """Randomly changes brightness, contrast, saturation, and hue."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = transforms.ColorJitter(
            brightness=brightness, 
            contrast=contrast, 
            saturation=saturation, 
            hue=hue
        )

    def __call__(self, sample):
        image = sample['image']
        pil_img = Image.fromarray(util.img_as_ubyte(image))
        image = util.img_as_float32(np.asarray(self.jitter(pil_img)))
        return {'image': image, 'label': sample['label']}


# Scikit-Image Transformations
class SkResize:
    """Resizes the image to a target output size (height, width)."""
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        resized = transform.resize(
            image, 
            self.output_size, 
            anti_aliasing=True, 
            mode='reflect'
        )
        return {'image': resized.astype(np.float32), 'label': sample['label']}


class SkAddGaussianNoise:
    """Adds zero-mean Gaussian noise for robustness against sensor noise."""
    def __init__(self, var=0.01):
        self.var = var

    def __call__(self, sample):
        image = sample['image']
        noisy = util.random_noise(image, mode='gaussian', var=self.var)
        return {'image': noisy.astype(np.float32), 'label': sample['label']}


class SkAdjustGamma:
    """Performs Gamma Correction for illumination augmentation."""
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range

    def __call__(self, sample):
        image = sample['image']
        gamma = random.uniform(*self.gamma_range)
        adjusted = exposure.adjust_gamma(image, gamma=gamma)
        return {'image': adjusted.astype(np.float32), 'label': sample['label']}

##### DATASET AND DATALOADERS

class FireRiskDataset(Dataset):
    """Read RGB images and integer labels using a CSV index.

    Args:
        csv_file: CSV with columns 'id' and 'label'.
        root_dir: Dataset directory containing the 'images' subdirectory.
        transform: Optional callable that receives and returns a sample dictionary.
        maxSize: Maximum subset size; 0 selects the full split.
    """

    def __init__(self, csv_file, root_dir, transform=None, maxSize=0):

        if not isinstance(maxSize, numbers.Integral) or maxSize < 0:
            raise ValueError('maxSize must be a nonnegative integer.')

        self.dataset = pd.read_csv(csv_file, dtype={'id': str, 'label': int})

        if not {'id', 'label'}.issubset(self.dataset.columns) or self.dataset.empty:
            raise ValueError('The CSV must contain nonempty id and label columns.')
        if not self.dataset['label'].isin(range(-1, NUM_CLASSES)).all():
            raise ValueError('Labels must be -1 (hidden test labels) or 0 through 6.')
        if maxSize > 0:
            indices = np.random.RandomState(SEED).permutation(len(self.dataset))
            self.dataset = self.dataset.iloc[indices[:maxSize]].reset_index(drop=True)

        self.img_dir = Path(root_dir) / 'images'
        self.transform = transform
        self.classes = CLASS_NAMES.copy()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.dataset.iloc[int(idx)]
        image = util.img_as_float32(io.imread(self.img_dir / f"{row['id']}.jpg"))

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected an RGB image with three channels.')
        sample = {'image': image, 'label': int(row['label'])}

        return self.transform(sample) if self.transform is not None else sample

def compute_class_weights(train_dataset):
    """Computes Inverse Class Frequency weights as a PyTorch Tensor."""
    train_df = train_dataset.dataset
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = len(train_df)
    
    weights = total_samples / (NUM_CLASSES * class_counts.astype(np.float32))
    print(f"Class distribution: {class_counts}")
    print(f"Class loss weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32).to(device)

def get_dataloaders(batch_size=64, img_size=308):
    """Images are 320x320, crops them into 308 and 308"""
    train_transform = Compose([
        RandomCrop(img_size),
        TVRandomHorizontalFlip(p=0.5),
        TVColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        TVRandomRotation(degrees=180),
        ToTensor(),
        Normalize(PIXEL_MEAN, PIXEL_STD),
    ])
    
    eval_transform = Compose([
        CenterCrop(img_size),
        ToTensor(),
        Normalize(PIXEL_MEAN, PIXEL_STD),
    ])
    
    train_ds = FireRiskDataset(DB_DIR / 'train_reduced.csv', DB_DIR, transform=train_transform)
    val_ds = FireRiskDataset(DB_DIR / 'val_reduced.csv', DB_DIR, transform=eval_transform)
    test_ds = FireRiskDataset(DB_DIR / 'test_reduced.csv', DB_DIR, transform=eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_ds

##### TRAINING
# Training model function
def train_model(model, criterion, optimizer, scheduler,
                num_epochs=25, plot_confusion_matrix=True):
    """Run fixed-budget training and restore the best validation-macro-F1 model."""

    if not isinstance(num_epochs, numbers.Integral) or num_epochs < 1:
        raise ValueError('num_epochs must be a positive integer.')
    if not isinstance(criterion, nn.CrossEntropyLoss) or criterion.reduction != 'mean':
        raise ValueError('Use CrossEntropyLoss with reduction="mean".')
    if any(len(dataloaders[phase].dataset) == 0 for phase in ('train', 'val')):
        raise ValueError('Training and validation datasets must be nonempty.')

    seed_everything()

    if dataloaders['train'].generator is not None:
        dataloaders['train'].generator.manual_seed(SEED)

    criterion = criterion.to(device)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -float('inf')
    best_predictions = best_labels = None
    labels_order = np.arange(NUM_CLASSES)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ('train', 'val'):
            model.train(phase == 'train')
            running_loss = 0.0
            loss_denominator = 0.0
            phase_predictions, phase_labels = [], []

            for sample in dataloaders[phase]:
                inputs = sample['image'].to(device, dtype=torch.float32)
                labels = sample['label'].to(device, dtype=torch.long)

                if phase == 'train':
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    if logits.shape != (labels.numel(), NUM_CLASSES):
                        raise ValueError('The network must return N x 7 logits.')
                    loss = criterion(logits, labels)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite loss; check inputs and optimiser settings.')
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                denominator = (labels.numel() if criterion.weight is None else
                               criterion.weight[labels].sum().item())
                running_loss += loss.item() * denominator
                loss_denominator += denominator
                phase_predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
                phase_labels.append(labels.detach().cpu().numpy())

            if phase == 'train' and scheduler is not None:
                scheduler.step()
            predictions = np.concatenate(phase_predictions)
            true_labels = np.concatenate(phase_labels)
            epoch_loss = running_loss / loss_denominator
            epoch_f1 = f1_score(true_labels, predictions, labels=labels_order,
                                average='macro', zero_division=0)
            print(f'{phase} Loss: {epoch_loss:.4f} Macro F1-score: {epoch_f1:.4f}')
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_predictions = predictions.copy()
                best_labels = true_labels.copy()
        print()

    model.load_state_dict(best_model_wts)
    model.eval()

    elapsed = time.time() - since
    print(f'Training complete in {int(elapsed // 60)}m {int(elapsed % 60)}s')
    print(f'Best validation macro F1-score: {best_f1:.6f}')

    if plot_confusion_matrix:
        cm = confusion_matrix(best_labels, best_predictions, labels=labels_order,
                              normalize='true')
        display = ConfusionMatrixDisplay(100 * cm, display_labels=CLASS_NAMES)
        display.plot(xticks_rotation='vertical', values_format='.1f')
        plt.title('Row-normalised confusion matrix (%)')
        plt.tight_layout()
        plt.show()

    return model