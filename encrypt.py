# File: encrypt.py
import numpy as np
from keygen import ChaoticMap, logistic_map, sine_map, quadratic_map, tent_map
from dna import DNAEncoder
from permute import build_permutation

# DNA XOR truth table (from paper Table 2)
DNA_XOR = {
    ('A','A'):'A',('A','C'):'C',('A','G'):'G',('A','T'):'T',
    ('C','A'):'C',('C','C'):'A',('C','G'):'T',('C','T'):'G',
    ('G','A'):'G',('G','C'):'T',('G','G'):'A',('G','T'):'C',
    ('T','A'):'T',('T','C'):'G',('T','G'):'C',('T','T'):'A',
}

class ImageEncryptor:
    def __init__(self, seeds, params, warmup=1000):
        self.seeds = seeds
        self.params = params
        self.cm_q = ChaoticMap(quadratic_map, seeds['q'], params['a'], warmup)
        self.cm_t = ChaoticMap(tent_map, seeds['t'], params['mu_t'], warmup)
        self.cm_log = ChaoticMap(logistic_map, seeds['log'], params['b'], warmup)
        self.cm_sin = ChaoticMap(sine_map, seeds['sin'], params['r_sin'], warmup)
        self.cm_sel = ChaoticMap(logistic_map, seeds['sel'], params['b_sel'], warmup)
        self.dna = DNAEncoder()

    def encrypt_channel(self, ch):
        H,W = ch.shape; N = H*W
        # XOR diffusion keys
        D1 = np.floor(self.cm_q.generate(N)*255).astype(np.uint8)
        D2 = np.floor(self.cm_t.generate(N)*255).astype(np.uint8)
        flat = ch.flatten().astype(np.uint8)
        x1 = flat ^ D1
        x2 = x1 ^ D2
        # DNA substitution
        R = np.floor(self.cm_log.generate(N)*8).astype(int)%8
        dna_plain = self.dna.encode(x2, R)
        # DNA-XOR with sine keystream
        dna_key = self.dna.encode(np.floor(self.cm_sin.generate(N)*255).astype(int), R)
        # element-wise DNA XOR across 4 nucleotides
        dna_xor = [ ''.join(DNA_XOR[(p[i], k[i])] for i in range(4))
                    for p,k in zip(dna_plain, dna_key) ]
        # decode with fixed codebook rule 3
        R_fixed = np.full(N, 2, dtype=int)
        x3 = self.dna.decode(dna_xor, R_fixed)
        # second XOR diffusion
        x4 = x3 ^ D1
        # block-wise scrambling...
        img4 = x4.reshape((H,W))
        h2,w2 = H//2, W//2
        blocks = [img4[i*h2:(i+1)*h2, j*w2:(j+1)*w2]
                  for i in range(2) for j in range(2)]
        sel = np.floor(self.cm_sel.generate(4)*4).astype(int)%4
        maps = [quadratic_map, tent_map, logistic_map, sine_map]
        out_blocks=[]
        for b, s in zip(blocks, sel):
            cm_block = ChaoticMap(maps[s], self.seeds['blk'], self.params['p'][s], warmup=0)
            flatb = b.flatten()
            perm = build_permutation(cm_block.generate(flatb.size))
            scrambled = flatb[perm]
            out_blocks.append(scrambled.reshape(b.shape))
        top = np.hstack(out_blocks[:2]); bot = np.hstack(out_blocks[2:])
        img5 = np.vstack((top,bot))
        return img5

    def encrypt(self, img):
        return np.stack([self.encrypt_channel(img[:,:,i]) for i in range(3)],axis=2)