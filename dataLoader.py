import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from scipy.signal import butter, lfilter

class AudioDatasetForViT(Dataset):
    def __init__(self, ai_directory, human_directory, output_directory, sr=16000, duration=3, augment=True):
        self.ai_files = glob.glob(os.path.join(ai_directory, '*.mp3'))
        self.human_files = glob.glob(os.path.join(human_directory, '*.mp3'))

        self.all_files = self.ai_files + self.human_files
        self.labels = [0] * len(self.ai_files) + [1] * len(self.human_files)
        self.output_directory = output_directory
        self.sr = sr
        self.duration = duration
        self.augment = augment

        # Hardcoded global mean and std values
        self.global_mean = -58.18715250929163
        self.global_std = 15.877255962380845 

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        while True:
            audio_path = self.all_files[idx]
            label = self.labels[idx]
            try:
                y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
                y = librosa.util.fix_length(y, size=self.sr * self.duration)

                if self.augment:
                    if random.random() < 0.05:
                        y = self.apply_augmentation(y)

                # Clamp the audio signal to avoid large values
                y = np.clip(y, -1.0, 1.0)

                # Compute the STFT
                S = np.abs(librosa.stft(y))**2

                # Add a small constant to avoid log of zero
                S_db = librosa.power_to_db(S + 1e-10, ref=np.max)

                # Normalize using hardcoded global mean and std
                S_db = (S_db - self.global_mean) / self.global_std

                # Ensure consistent dimensions (e.g., 1025 x 94)
                target_shape = (1025, 94)
                if S_db.shape != target_shape:
                    S_db = np.pad(S_db, (
                        (0, max(0, target_shape[0] - S_db.shape[0])), 
                        (0, max(0, target_shape[1] - S_db.shape[1]))
                    ), mode='constant', constant_values=-self.global_mean)
                    S_db = S_db[:target_shape[0], :target_shape[1]]

                spectrogram_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
                
                # Save spectrogram and label
                spectrogram_path = os.path.join(self.output_directory, f"{idx}.npy")
                np.save(spectrogram_path, spectrogram_tensor.numpy())
                label_path = os.path.join(self.output_directory, f"{idx}_label.npy")
                np.save(label_path, np.array(label))

                return spectrogram_tensor, label, audio_path
            except Exception as e:
                print(f"Skipping file {audio_path} due to error: {e}")
                idx = (idx + 1) % len(self.all_files)

    def apply_augmentation(self, y):
        if random.random() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y=y, rate=rate)

        if random.random() < 0.5:
            steps = np.random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)

        if random.random() < 0.5:
            noise_amp = 0.005 * np.random.uniform() * np.amax(y)
            y = y + noise_amp * np.random.normal(size=y.shape[0])

        if random.random() < 0.5:
            shift = np.random.randint(self.sr * self.duration)
            y = np.roll(y, shift)

        if random.random() < 0.5:
            y = np.flip(y)
        
        if random.random() < 0.5:
            y = self.apply_equalizer(y)
        
        return y

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def apply_equalizer(self, y):
        # Define frequency bands (in Hz)
        bands = [
            (20, 300),   # Bass
            (300, 2000), # Mid
            (2000, 8000) # Treble
        ]
       
        # Apply gain to each band
        for lowcut, highcut in bands:
            # Ensure the band frequencies are within the valid range
            if lowcut < self.sr / 2 and highcut < self.sr / 2:
                gain = np.random.uniform(0.5, 1.5) # Random gain between 0.5 and 1.5
                y = self.bandpass_filter(y, lowcut, highcut, self.sr) * gain
            
        return y
