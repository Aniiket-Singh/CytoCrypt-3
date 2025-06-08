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
        # 1) Generate diffusion sequence D
        D = np.floor(self.map_gen.generate_sequence(N, scale=(0, 255))).astype(np.uint8)
        # 2) Generate per-pixel DNA rule indices (0-7)
        raw_rules = self.map_gen.generate_sequence(N)
        R_seq = np.clip((raw_rules * 8).astype(int), 0, 7)
        # 3) First diffusion (XOR)
        flat = channel.flatten().astype(np.uint8)
        diff = flat ^ D
        # 4) DNA encode
        dna = self.dna.encode_channel(diff, R_seq)
        # 5) Scramble DNA list
        S = self.map_gen.generate_sequence(N)
        perm = build_permutation(S)
        dna_scr = [dna[i] for i in perm]
        # 6) Unscramble DNA back to original positions
        inv_perm = invert_permutation(perm)
        dna_unscr = [dna_scr[inv_perm[i]] for i in range(N)]
        # 7) Decode DNA with correct rules
        dec_flat = self.dna.decode_channel(dna_unscr, R_seq)
        # 8) Second diffusion (XOR)
        out_flat = dec_flat ^ D
        return out_flat.reshape((H, W))

    def encrypt(self, img):
        chans = [self.encrypt_channel(img[:, :, i]) for i in range(3)]
        return np.stack(chans, axis=2)