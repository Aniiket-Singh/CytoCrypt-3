# File: permute.py
import numpy as np

def build_permutation(seq):
    return np.argsort(seq)

def invert_permutation(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv
