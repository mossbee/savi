import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, training

# Parameters
data_dir = 'data/my_faces'  # Original dataset location
cropped_dir = data_dir + '_cropped'  # Directory to save cropped faces
batch_size = 32
workers = 0 if os.name == 'nt' else 8

# Initialize device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Create directory for cropped faces
os.makedirs(cropped_dir, exist_ok=True)

# Prepare dataset for face extraction
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, cropped_dir))
    for p, _ in dataset.samples
]

# Create data loader for face extraction
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

# Extract and save faces
print('Extracting faces...')
for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print(f'\rBatch {i+1} of {len(loader)}', end='')
print('\nFace extraction complete')

# Free up GPU memory
del mtcnn