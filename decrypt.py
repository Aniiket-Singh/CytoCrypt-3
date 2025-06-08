# File: decrypt.py
from encrypt import ImageEncryptor

def decrypt_image(img, seed, mu=3.99, warmup=1000):
    # symmetric: encryption pipeline is its own inverse
    return ImageEncryptor(seed, mu, warmup).encrypt(img)