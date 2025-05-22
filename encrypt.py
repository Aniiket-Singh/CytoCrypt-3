# File: encrypt.py
import numpy as np
from keygen import ChaoticMapGenerator
from dna import DNAEncoder
from permute import build_permutation, invert_permutation, apply_permutation

class ImageEncryptor:
    def __init__(self, seed, mu=3.99, warmup=1000):
        self.map_gen = ChaoticMapGenerator({'x0': seed}, {'mu': mu}, warmup)
        self.dna = DNAEncoder()

    def encrypt_channel(self, channel):
        H, W = channel.shape
        N = H * W
        # 1) Diffusion key
        D = np.floor(self.map_gen.generate_sequence(N, scale=(0, 255))).astype(np.uint8)
        # 2) DNA rule indices
        raw_rules = self.map_gen.generate_sequence(N)
        R_seq = np.clip((raw_rules * 8).astype(int), 0, 7)
        # 3) First XOR diffusion
        flat = channel.flatten().astype(np.uint8)
        diff = flat ^ D
        # 4) Dynamic DNA encode
        dna = self.dna.encode_channel(diff, R_seq)
        # 5) Scramble DNA order
        S = self.map_gen.generate_sequence(N)
        perm = build_permutation(S)
        dna_scr = [dna[i] for i in perm]
        # 6) Decode scrambled DNA to pixel values
        dec_scr = self.dna.decode_channel(dna_scr, R_seq)
        # 7) Unscramble pixel order
        inv_perm = invert_permutation(perm)
        unscr = apply_permutation(dec_scr, inv_perm)
        # 8) Second XOR diffusion
        out_flat = unscr ^ D
        return out_flat.reshape((H, W))

    def encrypt(self, img):
        return np.stack([self.encrypt_channel(img[:,:,i]) for i in range(3)], axis=2)