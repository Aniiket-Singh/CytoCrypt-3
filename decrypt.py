# File: decrypt.py
import numpy as np
from keygen import ChaoticMap, logistic_map, sine_map, quadratic_map, pwlcm_map
from dna import DNAEncoder
from permute import build_permutation, invert_permutation
from encrypt import DNA_XOR

class ImageDecryptor:
    def __init__(self, seeds, params, warmup=1000):
        self.seeds = seeds
        self.params = params
        self.warmup = warmup
        self.cm_log = ChaoticMap(logistic_map, seeds['log'], params['b'], warmup)
        self.cm_sin = ChaoticMap(sine_map,     seeds['sin'], params['r_sin'], warmup)
        self.cm_sel = ChaoticMap(logistic_map, seeds['sel'], params['b_sel'], warmup)
        self.dna = DNAEncoder()

    def decrypt_channel(self, ch):
        """Decrypt one grayscale channel"""
        H, W = ch.shape
        N = H * W
        # 1) Inverse permutation
        img = ch.reshape(H, W)
        h2, w2 = H//2, W//2
        blocks = [ img[i*h2:(i+1)*h2, j*w2:(j+1)*w2]
                   for i in range(2) for j in range(2) ]
        sel = (self.cm_sel.generate(4)*4).astype(int) % 4
        cmap_list = [quadratic_map, pwlcm_map, logistic_map, sine_map]
        unscr = []
        for block, idx in zip(blocks, sel):
            cm_b = ChaoticMap(cmap_list[idx], self.seeds['blk'], self.params['p'][idx], warmup=0)  # use self.seeds/self.params
            flatb = block.flatten()
            perm = build_permutation(cm_b.generate(flatb.size))
            invp = invert_permutation(perm)
            unscr.append(flatb[invp].reshape(block.shape))
        top = np.hstack(unscr[:2]); bot = np.hstack(unscr[2:])
        x = np.vstack((top, bot)).flatten().astype(np.uint8)

        # 2) Inverse diffusion layer 1
        cm_q = ChaoticMap(quadratic_map, self.seeds['q'], self.params['a'], self.warmup)
        D1 = (cm_q.generate(N)*255).astype(np.uint8)
        x = x ^ D1

        # 3) Inverse DNA stage
        R = (self.cm_log.generate(N)*8).astype(int)%8
        dna_fixed = self.dna.encode(x, np.full(N,2))
        dna_key   = self.dna.encode((self.cm_sin.generate(N)*255).astype(int), R)
        dna_plain = [ ''.join(DNA_XOR[(d,k)] for d,k in zip(df, dk))
                      for df, dk in zip(dna_fixed, dna_key) ]
        x = self.dna.decode(dna_plain, R)

        # 4) Inverse diffusion layer 2
        cm_t = ChaoticMap(pwlcm_map, self.seeds['t'], self.params['p_t'], self.warmup)
        D2 = (cm_t.generate(N)*255).astype(np.uint8)
        x = x ^ D2

        return x.reshape(H, W)

    def decrypt(self, img):
        return np.stack([self.decrypt_channel(img[:,:,i]) for i in range(3)], axis=2)
