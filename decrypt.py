# File: decrypt.py
import numpy as np
from keygen import ChaoticMap, logistic_map, sine_map, quadratic_map, tent_map
from dna import DNAEncoder
from permute import build_permutation, invert_permutation

class ImageDecryptor:
    def __init__(self, seeds, params, warmup=1000):
        self.seeds = seeds
        self.params = params
        self.cm_q = ChaoticMap(quadratic_map, seeds['q'], params['a'], warmup)
        self.cm_t = ChaoticMap(tent_map, seeds['t'], params['mu_t'], warmup)
        self.cm_log = ChaoticMap(logistic_map, seeds['log'], params['b'], warmup)
        self.cm_sin = ChaoticMap(sine_map, seeds['sin'], params['r_sin'], warmup)
        self.cm_sel = ChaoticMap(logistic_map, seeds['sel'], params['b_sel'], warmup)
        self.dna = DNAEncoder()

    def decrypt_channel(self, ch):
        H,W = ch.shape; N = H*W
        img4 = ch.reshape((H,W))
        h2,w2 = H//2, W//2
        blocks = [img4[i*h2:(i+1)*h2, j*w2:(j+1)*w2] for i in range(2) for j in range(2)]
        sel = np.floor(self.cm_sel.generate(4)*4).astype(int)%4
        maps = [quadratic_map, tent_map, logistic_map, sine_map]
        unscr_blocks=[]
        for b, s in zip(blocks, sel):
            cm_block = ChaoticMap(maps[s], self.seeds['blk'], self.params['p'][s], warmup=0)
            flatb = b.flatten()
            perm = build_permutation(cm_block.generate(flatb.size))
            invp = invert_permutation(perm)
            unscr = flatb[invp]
            unscr_blocks.append(unscr.reshape(b.shape))
        top = np.hstack(unscr_blocks[:2]); bot = np.hstack(unscr_blocks[2:])
        x4 = np.vstack((top, bot)).flatten().astype(np.uint8)
        D1 = np.floor(self.cm_q.generate(N)*255).astype(np.uint8)
        x3 = x4 ^ D1
        R = np.floor(self.cm_log.generate(N)*8).astype(int)%8
        dna_fixed = self.dna.encode(x3, np.full(N,2))
        dna_key = self.dna.encode(np.floor(self.cm_sin.generate(N)*255).astype(int), R)
        dna_plain = [ ''.join(DNAEncoder.RULES[r][pair]) for pair,r in zip(dna_fixed, R) ]
        x2 = self.dna.decode(dna_plain, R)
        D2 = np.floor(self.cm_t.generate(N)*255).astype(np.uint8)
        x1 = x2 ^ D2
        flat = x1 ^ D1
        return flat.reshape((H,W))

    def decrypt(self, img):
        return np.stack([self.decrypt_channel(img[:,:,i]) for i in range(3)], axis=2)