# File: decrypt.py
from encrypt import ImageEncryptor

def decrypt_image(img, seed, mu=3.99, warmup=1000):
    # symmetric: same as encrypt
    return ImageEncryptor(seed, mu, warmup).encrypt(img)