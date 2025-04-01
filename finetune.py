import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import numpy as np

# Parameters
cropped_dir = 'data/my_faces_cropped'  # Directory with cropped faces
output_dir = 'models'  # Directory to save models
batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Initialize the Inception Resnet V1 model for fine-tuning
print('Initializing InceptionResnetV1 model...')
train_dataset = datasets.ImageFolder(cropped_dir, transform=transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
]))
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(train_dataset.class_to_idx)
).to(device)

# Define optimizer and learning rate scheduler
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

# Split indices for training and validation
img_inds = np.arange(len(train_dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

# Create data loaders for training and validation
train_loader = DataLoader(
    train_dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    train_dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

# Define loss function and metrics
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# Initialize TensorBoard writer
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

# Initial evaluation
print('\nInitial evaluation')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

# Training loop
best_acc = 0.0
for epoch in range(epochs):
    print(f'\nEpoch {epoch+1}/{epochs}')
    print('-' * 10)

    # Train
    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    # Evaluate
    resnet.eval()
    val_loss, val_metrics = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    
    # Save checkpoint if accuracy improved
    if val_metrics['acc'] > best_acc:
        best_acc = val_metrics['acc']
        checkpoint = {
            'model_state_dict': resnet.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
            'epoch': epoch,
            'accuracy': val_metrics['acc']
        }
        torch.save(checkpoint, os.path.join(output_dir, 'facenet_best.pth'))
        print(f"Saved model with accuracy: {val_metrics['acc']:.4f}")

# Save final model
checkpoint = {
    'model_state_dict': resnet.state_dict(),
    'class_to_idx': train_dataset.class_to_idx,
    'epoch': epochs,
    'accuracy': val_metrics['acc']
}
torch.save(checkpoint, os.path.join(output_dir, 'facenet_final.pth'))
print(f"Training complete. Final model saved with accuracy: {val_metrics['acc']:.4f}")

# Close TensorBoard writer
writer.close()