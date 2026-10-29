# Creates the custom net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from auxiliary import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything()

# Following block squeezes branches with a parallel expansions
# This drastically lowers total parameter count and operations
# Also allows for more activation and normalization layers without growing paramter counts  
class SqueezeExpandBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze_a = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand_b = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1) # 1x1 channels, pointwise information
        self.expand_c = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1) # 3x3 channels
        
        out_channels = expand1x1_channels + expand3x3_channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_a(x)
        out_b = self.expand_b(x)
        out_c = self.expand_c(x)
        out = torch.cat([out_b, out_c], dim=1)
        out = self.bn(out)
        return self.act(out)

# CustomNet
# Same as the one implemented in the notebook, should adjust later
class CustomNet(nn.Module):
    """Implement a CNN mapping N x 3 x 224 x 224 to N x 7 logits."""

    def __init__(self, num_classes = NUM_CLASSES):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3) # 96 7x7 convolutional filters, this channels will be later reduced
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Squeeze-expand block
        self.fire2 = SqueezeExpandBlock(in_channels=96, squeeze_channels=16, expand1x1_channels=96, expand3x3_channels=32)
        self.fire3 = SqueezeExpandBlock(in_channels=128, squeeze_channels=32, expand1x1_channels=128, expand3x3_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fire4 =SqueezeExpandBlock(in_channels=192, squeeze_channels=48, expand1x1_channels=160, expand3x3_channels=96)

        # Adaptive pooling for variable crop sizes, avoids overfitting, although there is a big collapse we might have to adjust
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 

        # Fully connected layers
        self.fc1 = nn.Linear(256, 120)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        # Connect the layers you define in __init__ using this reference graph.
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool2(x)
        x = self.fire4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Loading data
train_loader, val_loader, test_loader, train_dataset = get_dataloaders(batch_size=128, img_size=308)
class_weights = compute_class_weights(train_dataset).to(device)

# Initializing the model
model_custom = CustomNet().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer_ft = optim.AdamW(
    filter(lambda p: p.requires_grad, model_custom.parameters()), 
    lr=FT_LR, 
    weight_decay=1e-4
)

exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, 
    mode='min', 
    factor=0.5, 
    patience=4
)

trained_custom = train_model_es(
    model=model_custom,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    patience=8,
    plot_confusion_matrix=True
)


# Generating the prediction
def generate_predictions(model, test_loader, output_path):
    """Computes softmax probabilities, validates shape and constraints, and saves a headerless CSV."""
    model.eval()
    batches = []

    with torch.inference_mode():
        for sample in test_loader:
            inputs = sample['image'].to(device, dtype=torch.float32)
            logits = model(inputs)
            if logits.shape != (inputs.shape[0], NUM_CLASSES):
                raise ValueError(f'The network must return N x {NUM_CLASSES} logits.')
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    if not batches:
        raise ValueError('The test split must be nonempty.')

    probabilities = np.concatenate(batches).astype(np.float32, copy=False)

    # Runtime validation
    expected_shape = (len(test_loader.dataset), NUM_CLASSES)
    if probabilities.shape != expected_shape:
        raise ValueError(f'Expected prediction shape {expected_shape}, got {probabilities.shape}.')
    if not np.isfinite(probabilities).all():
        raise ValueError('Submission probabilities must be finite.')
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Submission probabilities must lie in [0, 1].')
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError('Each probability row must sum to 1.')

    # Save headerless matrix of floats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, probabilities, delimiter=',', fmt='%.9g')
    print(f'Saved {output_path.name}: {probabilities.shape}')
    
    return probabilities

outputs_ft = generate_predictions(trained_custom, test_loader, OUTPUT_DIR / 'output_custom.csv')