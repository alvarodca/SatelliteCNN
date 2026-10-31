# Creates the custom net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from auxiliary import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything()

class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return self.act(y + self.skip(x))

# CustomNet
# Same as the one implemented in the notebook, should adjust later
class CustomNet(nn.Module):
    """Implement a CNN mapping N x 3 x 224 x 224 to N x 7 logits."""

    def __init__(self, num_classes = NUM_CLASSES,widths=(32, 64, 128, 256)):
        super().__init__()
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True)
        )

        #4 residual layers
        layers = []
        cin = widths[0]
        for i, cout in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers.append(ResBlock(cin, cout, stride=stride))
            layers.append(ResBlock(cout, cout, stride=1))
            cin = cout

        self.stages = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)

        # Lienar PRojection
        self.fc = nn.Linear(2 * widths[-1], num_classes)

    def forward(self, x):
        x = self.stages(self.stem(x))
        # Both average and max pooling
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(self.drop(pooled))




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


if __name__=="__main__":
    seed_everything()

    # Loading data
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(batch_size=128, img_size=308)
    class_weights = compute_softened_class_weights(train_dataset)

    # Initializing the model
    model_custom = CustomNet().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer_custom = optim.AdamW(model_custom.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler_custom = lr_scheduler.CosineAnnealingLR(optimizer_custom, T_max=80, eta_min=1e-5)

    trained_custom = train_model_es(
    model=model_custom,
    criterion=criterion,
    optimizer=optimizer_custom,
    scheduler=scheduler_custom,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    patience=8,
    plot_confusion_matrix=True
    )

    outputs_ft = generate_predictions(trained_custom, test_loader, OUTPUT_DIR / 'output_custom.csv')
