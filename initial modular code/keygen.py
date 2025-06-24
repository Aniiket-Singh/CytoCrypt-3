# Project: Dynamic DNA & Chaotic Map Image Encryption

# File: keygen.py
import numpy as np

class ChaoticMap:
    def __init__(self, func, seed, param, warmup=1000):
        """
        func: chaotic map function(x, param)
        seed: initial state x0
        param: control parameter
        warmup: iterations to discard
        """
        self.func = func
        self.x = seed
        self.param = param
        # Warm-up to remove transient effects
        for _ in range(warmup):
            self.x = self.func(self.x, self.param)

    def generate(self, length):
        """
        Generate `length` chaotic values, normalized to [0,1]
        """
        seq = np.empty(length)
        for i in range(length):
            self.x = self.func(self.x, self.param)
            seq[i] = self.x
        min_val, max_val = seq.min(), seq.max()
        if max_val - min_val == 0:
            return np.zeros_like(seq)
        return (seq - min_val) / (max_val - min_val)

# Chaotic map functions

def logistic_map(x, r):
    return r * x * (1 - x)

def sine_map(x, r):
    return r * np.sin(np.pi * x)

def quadratic_map(x, a):
    return a * x**2 + (1 - a) * x

def pwlcm_map(x, p):
    return x/p if x < p else (1 - x)/(1 - p)