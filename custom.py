# Creates the custom net
import torch
import torch.nn as nn
from auxiliary import *
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything()

# Data
train_loader, val_loader, test_loader, train_dataset = get_dataloaders(batch_size=64, img_size=308) # Automatically cropped
class_weights = compute_class_weights(train_dataset) # Class weights for clas imbalance 

# CustomNet
# Same as the one implemented in the notebook, should adjust later
class CustomNet(nn.Module):
    """Implement a CNN mapping N x 3 x 224 x 224 to N x 7 logits."""

    def __init__(self):
        super().__init__()
        # Define conv1, conv2, conv3, pool, fc1 and fc2 here.
        # Suggested channel progression: 3 -> 6 -> 16 -> 32.
        # Three 5x5 convolutions and 2x2 pools leave 24x24 spatial features.
        # Each convolution must have a distinct name: do not overwrite conv2.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 7)

    def forward(self, x):
        # Connect the layers you define in __init__ using this reference graph.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initializing the model
model_custom = CustomNet().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer_ft = optim.AdamW(
    filter(lambda p: p.requires_grad, model_custom.parameters()), 
    lr=FT_LR, 
    weight_decay=1e-4
)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

# Training loop
trained_custom = train_model(
    model=model_custom,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    num_epochs=10, # Should adjust
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