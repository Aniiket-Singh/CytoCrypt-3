# File: permute.py
import numpy as np

def build_permutation(seq):
    """Return indices that sort seq"""
    return np.argsort(seq)

def invert_permutation(perm):
    """Return inverse permutation"""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv
