# File: encrypt.py
import numpy as np
from keygen import ChaoticMapGenerator
from dna import DNAEncoder
from permute import build_permutation

class ImageEncryptor:
    def __init__(self, seed, mu=3.99, warmup=1000):
        self.map_gen = ChaoticMapGenerator({'x0': seed}, {'mu': mu}, warmup)
        self.dna = DNAEncoder()

    def encrypt_channel(self, channel):
        H, W = channel.shape; N = H*W
        # generate keystreams
        D = np.floor(self.map_gen.generate_sequence(N, (0,255))).astype(np.uint8)
        R = np.clip((self.map_gen.generate_sequence(N)*8).astype(int), 0, 7)
        S = self.map_gen.generate_sequence(N)
        # diffusion
        flat = channel.flatten().astype(np.uint8)
        diff = flat ^ D
        # confusion: dynamic DNA + permutation
        dna = self.dna.encode(diff, R)
        perm = build_permutation(S)
        dna_scr = [dna[i] for i in perm]
        # decode scrambled DNA back to pixel-level diff
        diff_scr = self.dna.decode(dna_scr, R)
        # final diffusion
        out_flat = diff_scr ^ D
        return out_flat.reshape((H, W))

    def encrypt(self, img_array):
        chans = [self.encrypt_channel(img_array[:,:,i]) for i in range(3)]
        return np.stack(chans, axis=2)