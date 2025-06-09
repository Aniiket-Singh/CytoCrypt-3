# Project: Dynamic DNA & Chaotic Map Image Encryption

# File: keygen.py
import numpy as np

class ChaoticMap:
    def __init__(self, func, seed, param, warmup=1000):
        self.func = func
        self.x = seed
        self.param = param
        for _ in range(warmup):
            self.x = self.func(self.x, self.param)

    def generate(self, length):
        seq = np.empty(length)
        for i in range(length):
            self.x = self.func(self.x, self.param)
            seq[i] = self.x
        # normalize to [0,1]
        min_val, max_val = seq.min(), seq.max()
        if max_val - min_val == 0:
            seq_norm = np.zeros_like(seq)
        else:
            seq_norm = (seq - min_val) / (max_val - min_val)
        return seq_norm

# Map functions

def logistic_map(x, r):
    return r * x * (1 - x)

def sine_map(x, r):
    return r * np.sin(np.pi * x)

def quadratic_map(x, a):
    return a * x**2 + (1 - a) * x

def tent_map(x, mu):
    return mu * x if x < 0.5 else mu * (1 - x)