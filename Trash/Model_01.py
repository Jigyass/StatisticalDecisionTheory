import os
import torch
import torchaudio
import random
import torchaudio.transforms as T

class AudioDataset:
    def __init__(self, lossless_dir, lossy_dir, segment_duration=0.1):
        """
        Initializes the dataset and processes songs one by one, adding valid pairs to the dataset.
        """
        self.lossless_files = sorted(
            [os.path.join(lossless_dir, f) for f in os.listdir(lossless_dir) if os.path.isfile(os.path.join(lossless_dir, f))]
        )
        self.lossy_files = sorted(
            [os.path.join(lossy_dir, f) for f in os.listdir(lossy_dir) if os.path.isfile(os.path.join(lossy_dir, f))]
        )

        assert len(self.lossless_files) == len(self.lossy_files), "Mismatch in number of lossless and lossy files!"

        self.segment_duration = segment_duration
        self.data = []  # Store valid segment pairs in memory

    def process_and_add(self):
        """
        Processes each song and adds valid pairs (with matching segment counts) to the dataset.
        """
        for idx, (lossless_path, lossy_path) in enumerate(zip(self.lossless_files, self.lossy_files)):
            song_data = self.process_pair(lossless_path, lossy_path)
            if song_data:  # Only add if the song pair is valid
                self.data.extend(song_data)
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(self.lossless_files)} songs...")

        print(f"Dataset created with {len(self.data)} valid segment pairs.")

    def process_pair(self, lossless_path, lossy_path):
        """
        Processes a pair of lossless and lossy files into aligned stereo segments.
        Excludes the pair if the number of segments is unequal.
        """
        lossless_segments, lossless_segment_size = self.preprocess(lossless_path)
        lossy_segments, lossy_segment_size = self.preprocess(lossy_path)

        # Exclude songs with unequal segment counts
        if len(lossless_segments) != len(lossy_segments):
            print(f"Skipping {lossless_path} and {lossy_path} due to unequal segments.")
            return []

        # Randomly pad lossy segments to match lossless size
        padded_lossy_segments = [
            self.random_pad(lossy_segment, lossless_segment.shape[1])
            for lossy_segment, lossless_segment in zip(lossy_segments, lossless_segments)
        ]

        return list(zip(padded_lossy_segments, lossless_segments))

    def preprocess(self, file_path):
        """
        Loads an audio file, calculates dynamic segment size, and splits into stereo segments.
        """
        waveform, sample_rate = torchaudio.load(file_path)

        # Calculate segment size dynamically
        segment_size = int(sample_rate * self.segment_duration)

        # Split waveform into fixed-size segments
        segments = [
            waveform[:, i:i + segment_size]
            for i in range(0, waveform.shape[1], segment_size)
            if waveform[:, i:i + segment_size].shape[1] == segment_size
        ]
        return segments, segment_size

    def random_pad(self, segment, target_size):
        """
        Randomly pads the input segment to match the target size.
        """
        current_size = segment.shape[1]
        if current_size >= target_size:
            return segment  # No padding needed

        # Calculate padding size
        padding_size = target_size - current_size
        front_pad = random.randint(0, padding_size)
        back_pad = padding_size - front_pad

        return torch.nn.functional.pad(segment, (front_pad, back_pad))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single pair of lossy and lossless stereo segments.
        """
        return self.data[idx]

lossless_dir = "/home/j597s263/scratch/j597s263/Datasets/Audio/Lossless/temp/"
lossy_dir = "/home/j597s263/scratch/j597s263/Datasets/Audio/Lossy/temp/"

# Create the dataset processor
dataset = AudioDataset(lossless_dir, lossy_dir, segment_duration=0.1)

# Process songs one by one
dataset.process_and_add()

# Check the size of the dataset
print(f"Total number of segment pairs: {len(dataset)}")


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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


# Training loop
def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10, device='cuda:0'):
    model.train()
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")        


# DataLoader
from torch.utils.data import DataLoader
audio_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model and training setup
device = 'cuda'
model = AudioEnhancer(num_transformer_layers=2, num_heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Loss function with dummy feature extractor
feature_extractor = DummyFeatureExtractor().to(device)
loss_fn = PerceptualLoss(feature_extractor).to(device)

# Train the model
train_model(model, audio_dataloader, optimizer, loss_fn, num_epochs=10, device=device)

# Path to save the entire model
model_save_path = '/home/j597s263/scratch/j597s263/Models/Lad_0.01.mod'

# Save the entire model
torch.save(model, model_save_path)
print(f"Entire model saved to {model_save_path}.")