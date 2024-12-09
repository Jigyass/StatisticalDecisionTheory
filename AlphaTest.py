#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchaudio.transforms as T
import torch
import torchaudio
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# In[2]:


class AudioDataset(Dataset):
    def __init__(self, lossless_dir, lossy_dir, segment_duration=1, target_sample_rate=44000):
        """
        Initializes the dataset by processing all audio files in the given directories.
        Each observation is a pair of corresponding lossy and lossless 1-second stereo segments.
        """
        self.data = []
        self.segment_duration = segment_duration
        self.target_sample_rate = target_sample_rate

        # Get sorted lists of lossless and lossy files
        lossless_files = sorted([os.path.join(lossless_dir, f) for f in os.listdir(lossless_dir)])
        lossy_files = sorted([os.path.join(lossy_dir, f) for f in os.listdir(lossy_dir)])

        # Ensure equal number of files
        assert len(lossless_files) == len(lossy_files), "Mismatch in number of lossless and lossy files!"

        # Process files and create dataset
        for lossless_file, lossy_file in zip(lossless_files, lossy_files):
            self.data.extend(self._process_pair(lossless_file, lossy_file))
        
        print(f"Dataset created with {len(self.data)} segment pairs.")

    def _process_pair(self, lossless_path, lossy_path):
        """
        Processes a pair of lossless and lossy files into aligned 1-second stereo segments.
        """
        lossless_segments = self._preprocess_and_split(lossless_path, resample_to=self.target_sample_rate)
        lossy_segments = self._preprocess_and_split(lossy_path, resample_to=self.target_sample_rate)
        assert len(lossless_segments) == len(lossy_segments), f"Segment mismatch in {lossless_path} and {lossy_path}!"
        return list(zip(lossy_segments, lossless_segments))

    def _preprocess_and_split(self, file_path, resample_to=None):
        """
        Loads an audio file, optionally resamples, and splits it into 1-second stereo segments.
        """
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if needed
        if resample_to and sample_rate != resample_to:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=resample_to)
            waveform = resampler(waveform)
        
        segment_length = int(resample_to if resample_to else sample_rate) * self.segment_duration  # Samples per segment
        return [waveform[:, i:i + segment_length] for i in range(0, waveform.shape[1], segment_length) 
                if waveform[:, i:i + segment_length].shape[1] == segment_length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[3]:


dataset = AudioDataset(lossless_dir='/home/j597s263/scratch/j597s263/Datasets/Audio/Lossless/', lossy_dir='/home/j597s263/scratch/j597s263/Datasets/Audio/Lossy/')

print(f"Total number of observations (segment pairs): {len(dataset)}")

# Example: Retrieve a single observation
lossy_segment, lossless_segment = dataset[0]
print(f"Lossy segment shape: {lossy_segment.shape}")
print(f"Lossless segment shape: {lossless_segment.shape}")


# In[4]:


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
            TransformerEncoderLayer(d_model=cnn_filters[-1], nhead=num_heads, dim_feedforward=512, activation='relu', batch_first=True),
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
        # Input: [batch_size, 2, 48000]
        
        # CNN Encoder
        x = self.encoder(x)  # Shape: [batch_size, cnn_filters[-1], 48000]
        
        # Permute for Transformer
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 48000, cnn_filters[-1]]
        x = self.transformer(x)  # Shape: [batch_size, 48000, cnn_filters[-1]]
        x = x.permute(0, 2, 1)  # Shape: [batch_size, cnn_filters[-1], 48000]
        
        # CNN Decoder
        x = self.decoder(x)  # Shape: [batch_size, 2, 48000]
        
        return x


# In[5]:


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


# In[6]:


def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10):
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


# In[7]:


audio = DataLoader(dataset, batch_size=8, shuffle=True)


# In[8]:


device = 'cuda'
model = AudioEnhancer(num_transformer_layers=4, num_heads=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)


# In[9]:


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

# Initialize feature extractor
feature_extractor = DummyFeatureExtractor().to(device)
loss_fn = PerceptualLoss(feature_extractor).to(device)


# In[10]:


# Train the model
train_model(model, audio, optimizer, loss_fn, num_epochs=1)


# In[ ]:


torch.save(model, '/home/j597s263/scratch/j597s263/Models/Audio/AlphaTest.mod')

