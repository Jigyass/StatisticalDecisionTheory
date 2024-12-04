import os
import torchaudio
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, lossless_dir, lossy_dir, segment_duration=0.1, target_sample_rate=44000):
        """
        Initializes the dataset and processes songs one by one, ensuring both lossy and lossless
        are resampled to 44kHz if needed.
        """
        self.lossless_files = sorted(
            [os.path.join(lossless_dir, f) for f in os.listdir(lossless_dir) if os.path.isfile(os.path.join(lossless_dir, f))]
        )
        self.lossy_files = sorted(
            [os.path.join(lossy_dir, f) for f in os.listdir(lossy_dir) if os.path.isfile(os.path.join(lossy_dir, f))]
        )

        assert len(self.lossless_files) == len(self.lossy_files), "Mismatch in number of lossless and lossy files!"

        self.segment_duration = segment_duration
        self.target_sample_rate = target_sample_rate
        self.data = []  # Store valid segment pairs in memory

        # Process and add all files
        self.process_and_add()

    def process_and_add(self):
        """
        Processes each song and adds valid segment pairs to the dataset.
        """
        for idx, (lossless_path, lossy_path) in enumerate(zip(self.lossless_files, self.lossy_files)):
            song_data = self.process_pair(lossless_path, lossy_path)
            if song_data:
                self.data.extend(song_data)
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(self.lossless_files)} songs...")

        print(f"Dataset created with {len(self.data)} valid segment pairs.")

    def process_pair(self, lossless_path, lossy_path):
        """
        Processes a pair of lossless and lossy files, resampling if necessary, and splitting into segments.
        """
        lossless_waveform, lossless_sr = torchaudio.load(lossless_path)
        lossy_waveform, lossy_sr = torchaudio.load(lossy_path)

        # Resample to target sample rate if needed
        if lossless_sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=lossless_sr, new_freq=self.target_sample_rate)
            lossless_waveform = resampler(lossless_waveform)
        if lossy_sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=lossy_sr, new_freq=self.target_sample_rate)
            lossy_waveform = resampler(lossy_waveform)

        # Segment duration in samples
        segment_size = int(self.target_sample_rate * self.segment_duration)

        # Split waveforms into segments
        lossless_segments = [
            lossless_waveform[:, i:i + segment_size]
            for i in range(0, lossless_waveform.shape[1], segment_size)
            if lossless_waveform[:, i:i + segment_size].shape[1] == segment_size
        ]
        lossy_segments = [
            lossy_waveform[:, i:i + segment_size]
            for i in range(0, lossy_waveform.shape[1], segment_size)
            if lossy_waveform[:, i:i + segment_size].shape[1] == segment_size
        ]

        # Ensure equal number of segments
        if len(lossless_segments) != len(lossy_segments):
            print(f"Skipping {lossless_path} and {lossy_path} due to unequal segments.")
            return []

        return list(zip(lossy_segments, lossless_segments))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single pair of lossy and lossless stereo segments.
        """
        return self.data[idx]

lossless_dir = "/home/j597s263/scratch/j597s263/Datasets/Audio/Lossless/"
lossy_dir = "/home/j597s263/scratch/j597s263/Datasets/Audio/Lossy/"

# Create the dataset processor
dataset = AudioDataset(lossless_dir, lossy_dir, segment_duration=0.1, target_sample_rate=44000)

print(f"Total segment pairs: {len(dataset)}")
lossy_segment, lossless_segment = dataset[0]
print(f"Lossy segment shape: {lossy_segment.shape}, Lossless segment shape: {lossless_segment.shape}")

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Verify batch shapes
for lossy, lossless in dataloader:
    print(f"Batch lossy shape: {lossy.shape}, Batch lossless shape: {lossless.shape}")
    break

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt


class AudioEnhancer(nn.Module):
    def __init__(self, num_transformer_layers=2, num_heads=8, cnn_filters=[32, 64, 128, 256]):
        super(AudioEnhancer, self).__init__()
        
        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(2, cnn_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_filters[2], cnn_filters[3], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Transformer
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=cnn_filters[-1], 
                nhead=num_heads, 
                dim_feedforward=512, 
                activation='relu', 
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )
        
        # CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(cnn_filters[3], cnn_filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(cnn_filters[2], cnn_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(cnn_filters[1], cnn_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(cnn_filters[0], 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Input: [batch_size, 2, 4800]
        
        # CNN Encoder
        x = self.encoder(x)  # Shape: [batch_size, cnn_filters[-1], 4800]
        
        # Permute for Transformer
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 4800, cnn_filters[-1]]
        x = self.transformer(x)  # Shape: [batch_size, 4800, cnn_filters[-1]]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, cnn_filters[-1], 4800]
        
        # CNN Decoder
        x = self.decoder(x)  # Shape: [batch_size, 2, 4800]
        
        return x



class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # Compute perceptual features
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        # Perceptual loss
        perceptual_loss = self.mse_loss(pred_features, target_features)
        
        # Reconstruction loss
        reconstruction_loss = self.mse_loss(pred, target)
        
        return perceptual_loss + reconstruction_loss


# Dummy feature extractor for perceptual loss
class DummyFeatureExtractor(nn.Module):
    def __init__(self):
        super(DummyFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.features(x)


import matplotlib.pyplot as plt

def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10, device='cuda:0', plot_path='/home/j597s263/scratch/j597s263/StatisticalDecisionTheory/Exp_4.png'):
    model.train()
    losses = []  # To store loss values for plotting

    for epoch in range(num_epochs):
        total_loss = 0
        for lossy, lossless in dataloader:
            # Move to GPU if available
            lossy, lossless = lossy.to(device), lossless.to(device)
            
            # Forward pass
            output = model(lossy)
            
            # Compute loss
            loss = loss_fn(output, lossless)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the loss curve as a plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_path)  # Save the plot to a file
    print(f"Training loss plot saved to {plot_path}")

    return losses

# Model and training setup
device = 'cuda'
model = AudioEnhancer(num_transformer_layers=2, num_heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Loss function with dummy feature extractor
feature_extractor = DummyFeatureExtractor().to(device)
loss_fn = PerceptualLoss(feature_extractor).to(device)

# Train the model
train_model(model, dataloader, optimizer, loss_fn, num_epochs=20, device=device)

# Path to save the entire model
model_save_path = '/home/j597s263/scratch/j597s263/Models/Exp_4.mod'

# Save the entire model
torch.save(model, model_save_path)
print(f"Entire model saved to {model_save_path}.")