# Creates the transfer learning model
import torch
import torch.nn as nn
from auxiliary import *
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_everything()

# Data augmentation
train_loader, val_loader, test_loader, train_dataset = get_dataloaders(batch_size=64, img_size=308) # Automatically cropped
class_weights = compute_class_weights(train_dataset) # Class weights for clas imbalance 

# DINOnet, simple linear probe
class DINONet(nn.Module):
    def __init__(self, num_classes = NUM_CLASSES, model_name="dinov2_vits14"):
        super().__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)

        embed_dim = self.backbone.embed_dim

    
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Linear probing, prediction head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# Initializing the model
model_ft = DINONet().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer_ft = optim.AdamW(
    filter(lambda p: p.requires_grad, model_ft.parameters()), 
    lr=FT_LR, 
    weight_decay=1e-4
)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

# Training loop
trained_dino = train_model(
    model=model_ft,
    criterion=criterion,
    optimizer=optimizer_ft,
    scheduler=exp_lr_scheduler,
    train_loader=train_loader,  
    val_loader=val_loader,     
    num_epochs=30,
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

outputs_ft = generate_predictions(trained_dino, test_loader, OUTPUT_DIR / 'output_ft.csv')