import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import random

class GoDataset(Dataset):
    def __init__(self, source_input, max_games=None):
        """
        Args:
            source_input: Can be a directory path (str) OR a list of file paths (list).
            max_games: Optional limit if source is a directory.
        """
        if isinstance(source_input, list):
            self.files = source_input
            print(f"   -> Initialized with specific file list ({len(self.files)} files).")
        else:
            print(f"🔍 Scanning {source_input}...")
            self.files = glob.glob(os.path.join(source_input, '*.npz'))
            if max_games and max_games < len(self.files):
                random.shuffle(self.files)
                self.files = self.files[:max_games]
        
        self.inputs = []
        self.targets = []
        
        # Load logic
        for i, filepath in enumerate(self.files):
            try:
                data = np.load(filepath)
                # Store as int8 to save RAM
                self.inputs.append(data['inputs'].astype(np.int8))
                self.targets.append(data['targets'])
            except Exception:
                pass

        if self.inputs:
            self.inputs = np.concatenate(self.inputs, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)
            # print(f"      Data Loaded: {len(self.inputs)} moves.")
        else:
            print("⚠️ Warning: No data loaded in this chunk.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_np = self.inputs[idx]
        target_np = self.targets[idx]
        
        input_tensor = torch.tensor(input_np, dtype=torch.float32)
        target_tensor = torch.tensor(target_np, dtype=torch.long)
        
        return input_tensor, target_tensor