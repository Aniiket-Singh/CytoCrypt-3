# File: decrypt.py
import numpy as np
from keygen import ChaoticMapGenerator
from dna import DNAEncoder
from permute import build_permutation, invert_permutation

class ImageDecryptor:
    def __init__(self, seed, mu=3.99, warmup=1000):
        self.map_gen = ChaoticMapGenerator({'x0': seed}, {'mu': mu}, warmup)
        self.dna = DNAEncoder()

    def decrypt_channel(self, channel):
        H, W = channel.shape; N = H*W
        # regenerate keystreams in same order
        D = np.floor(self.map_gen.generate_sequence(N, (0,255))).astype(np.uint8)
        R = np.clip((self.map_gen.generate_sequence(N)*8).astype(int), 0, 7)
        S = self.map_gen.generate_sequence(N)
        # inverse diffusion
        flat = channel.flatten().astype(np.uint8)
        diff_scr = flat ^ D
        # confusion inverse: encode now (to DNA), unscramble, decode
        dna_scr = self.dna.encode(diff_scr, R)
        perm = build_permutation(S)
        inv = invert_permutation(perm)
        dna = [dna_scr[inv[i]] for i in range(N)]
        diff = self.dna.decode(dna, R)
        # final diffusion inverse
        out_flat = diff ^ D
        return out_flat.reshape((H, W))

    def decrypt(self, img_array):
        chans = [self.decrypt_channel(img_array[:,:,i]) for i in range(3)]
        return np.stack(chans, axis=2)