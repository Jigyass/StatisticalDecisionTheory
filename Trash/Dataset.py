import os
import torch
import torchaudio
import random
import pickle
from torch.utils.data import Dataset
import torchaudio.transforms as T

class AudioDatasetBatch:
    def __init__(self, lossless_dir, lossy_dir, segment_duration=0.1, batch_size=500):
        """
        Initializes the batch-based dataset processor.
        """
        self.lossless_files = sorted(
            [os.path.join(lossless_dir, f) for f in os.listdir(lossless_dir) if os.path.isfile(os.path.join(lossless_dir, f))]
        )
        self.lossy_files = sorted(
            [os.path.join(lossy_dir, f) for f in os.listdir(lossy_dir) if os.path.isfile(os.path.join(lossy_dir, f))]
        )

        assert len(self.lossless_files) == len(self.lossy_files), "Mismatch in number of lossless and lossy files!"
        self.segment_duration = segment_duration
        self.batch_size = batch_size

    def process_pair(self, lossless_path, lossy_path):
        """
        Processes a pair of lossless and lossy files into aligned stereo segments.
        """
        lossless_segments, lossless_segment_size = self.preprocess(lossless_path)
        lossy_segments, lossy_segment_size = self.preprocess(lossy_path)

        # Handle floating-point mismatch in segment counts
        if len(lossless_segments) > len(lossy_segments):
            for _ in range(len(lossless_segments) - len(lossy_segments)):
                # Add padded segments to lossy segments to match lossless count
                padding_segment = torch.zeros_like(lossless_segments[0])
                lossy_segments.append(padding_segment)

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

    def process_and_save_batches(self, output_dir):
        """
        Processes the dataset in batches and saves each batch to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        batch_data = []
        batch_count = 0

        for idx, (lossless_file, lossy_file) in enumerate(zip(self.lossless_files, self.lossy_files)):
            batch_data.extend(self.process_pair(lossless_file, lossy_file))

            # Save batch if it exceeds batch_size
            while len(batch_data) >= self.batch_size:
                batch_file = os.path.join(output_dir, f"batch_{batch_count}.pkl")
                with open(batch_file, 'wb') as f:
                    pickle.dump(batch_data[:self.batch_size], f)
                print(f"Saved {self.batch_size} segments to {batch_file}.")
                batch_data = batch_data[self.batch_size:]  # Retain leftover data
                batch_count += 1

        # Save any remaining data
        if batch_data:
            batch_file = os.path.join(output_dir, f"batch_{batch_count}.pkl")
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_data, f)
            print(f"Saved {len(batch_data)} segments to {batch_file}.")

lossless_dir = '/home/j597s263/scratch/j597s263/Datasets/Audio/Lossless/'
lossy_dir = '/home/j597s263/scratch/j597s263/Datasets/Audio/Lossy/'
output_dir = '/home/j597s263/scratch/j597s263/Datasets/Audio/Batches/'

# Create the batch processor with 0.1-second segments
dataset_batch = AudioDatasetBatch(lossless_dir, lossy_dir, segment_duration=0.1, batch_size=100)

# Process and save batches
dataset_batch.process_and_save_batches(output_dir)