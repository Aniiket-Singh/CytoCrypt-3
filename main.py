# File: main.py
import sys
from PIL import Image
import numpy as np
from encrypt import ImageEncryptor
from decrypt import decrypt_image

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def save_image(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

if __name__ == '__main__':
    mode, inp, out, seed = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    img = load_image(inp)
    if mode == 'enc':
        result = ImageEncryptor(seed).encrypt(img)
    else:
        result = decrypt_image(img, seed)
    save_image(result, out)
    print(f"{mode.upper()} complete. Output: {out}")