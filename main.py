# File: main.py
import sys
from PIL import Image
import numpy as np
from encrypt import ImageEncryptor
from decrypt import ImageDecryptor

# Configuration loader
def load_config():
    seeds = {'q':0.345,'t':0.678,'log':0.123,'sin':0.901,'sel':0.456,'blk':0.789}
    params = {'a':1.5,'p_t':0.3,'b':3.9,'r_sin':0.9,'b_sel':3.9,'p':[0.3,0.5,0.7,0.9]}
    return seeds, params

# I/O helpers
def load_image(path): return np.array(Image.open(path).convert('RGB'))
def save_image(arr,path): Image.fromarray(arr.astype(np.uint8)).save(path)

if __name__=='__main__':
    if len(sys.argv)<4:
        print("Usage: python main.py <enc/dec> <in> <out>")
        sys.exit(1)
    mode, inp, out = sys.argv[1:4]
    seeds, params = load_config()
    img = load_image(inp)
    if mode=='enc': res = ImageEncryptor(seeds,params).encrypt(img)
    else:        res = ImageDecryptor(seeds,params).decrypt(img)
    save_image(res, out)
    print(f"{mode.upper()} complete: {out}")
