import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

# Parameters
workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

# Initialize MTCNN for face detection
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Initialize InceptionResnetV1 for embeddings
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)

# Load fine-tuned weights if available
fine_tuned_weights = 'models/facenet_best.pth'  # Path to fine-tuned weights
if os.path.exists(fine_tuned_weights):
    print(f'Loading fine-tuned weights from {fine_tuned_weights}')
    checkpoint = torch.load(fine_tuned_weights, map_location=device)
    resnet.load_state_dict(checkpoint['model_state_dict'])
else:
    print('Using pre-trained weights (no fine-tuned weights found)')

# Define dataset and data loader
data_dir = 'data/test_images'  # Path to test images
dataset = datasets.ImageFolder(data_dir)
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, num_workers=workers, collate_fn=lambda x: x[0])

# Perform face detection and extract embeddings
aligned = []
names = []
for img, label in loader:
    # Detect face and align
    aligned_face, prob = mtcnn(img, return_prob=True)
    if aligned_face is not None:
        print(f'Face detected with probability: {prob:.6f}')
        aligned.append(aligned_face)
        names.append(dataset.idx_to_class[label])

# Stack aligned faces and compute embeddings
if aligned:
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()
    print('Embeddings generated successfully.')
else:
    print('No faces detected.')

# Save embeddings and names for further use
output_file = 'embeddings.npz'
np.savez(output_file, embeddings=embeddings.numpy(), names=names)
print(f'Embeddings saved to {output_file}')