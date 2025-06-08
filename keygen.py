# Project: Dynamic DNA & Chaotic Map Image Encryption

# File: keygen.py
import numpy as np

class ChaoticMapGenerator:
    def __init__(self, seed_vals, map_params, warmup=1000):
        self.state = seed_vals.copy()
        self.params = map_params.copy()
        for _ in range(warmup):
            self._iterate_map()

    def _iterate_map(self):
        x = self.state['x0']
        mu = self.params.get('mu', 3.99)
        x = mu * x * (1 - x)
        x = mu * np.sin(np.pi * x)
        self.state['x0'] = x
        return x

    def generate_sequence(self, length, scale=(0, 1)):
        seq = np.empty(length)
        for i in range(length):
            seq[i] = self._iterate_map()
        low, high = scale
        seq_norm = (seq - seq.min()) / (seq.max() - seq.min())
        return low + (high - low) * seq_norm