# File: permute.py
import numpy as np

# Permutation utilities for scrambling

def build_permutation(seq):
    """Returns indices that would sort the sequence"""
    return np.argsort(seq)


def invert_permutation(perm):
    """Computes inverse of a permutation array"""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def apply_permutation(data, perm):
    """Applies permutation to data array"""
    return data[perm]