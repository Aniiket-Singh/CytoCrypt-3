# File: encrypt.py
import numpy as np
from keygen import ChaoticMap, logistic_map, sine_map, quadratic_map, pwlcm_map
from dna import DNAEncoder
from permute import build_permutation

# DNA XOR table (Table 2)
DNA_XOR = {
    ('A','A'):'A',('A','C'):'C',('A','G'):'G',('A','T'):'T',
    ('C','A'):'C',('C','C'):'A',('C','G'):'T',('C','T'):'G',
    ('G','A'):'G',('G','C'):'T',('G','G'):'A',('G','T'):'C',
    ('T','A'):'T',('T','C'):'G',('T','G'):'C',('T','T'):'A',
}

class ImageEncryptor:
    def __init__(self, seeds, params, warmup=1000):
        """
        seeds: dict of {'q','t','log','sin','sel','blk'}
        params: dict of {'a','p_t','b','r_sin','b_sel','p'}
        warmup: iterations to discard
        """
        self.seeds = seeds
        self.params = params
        self.warmup = warmup
        # Shared chaotic generators
        self.cm_log = ChaoticMap(logistic_map, seeds['log'], params['b'], warmup)
        self.cm_sin = ChaoticMap(sine_map,     seeds['sin'], params['r_sin'], warmup)
        self.cm_sel = ChaoticMap(logistic_map, seeds['sel'], params['b_sel'], warmup)
        self.dna = DNAEncoder()

    def encrypt_channel(self, ch):
        """Encrypt one grayscale channel"""
        H, W = ch.shape
        N = H * W
        # 1) Diffusion keys (fresh per channel)
        cm_q = ChaoticMap(quadratic_map, self.seeds['q'], self.params['a'], self.warmup)  # use self.seeds/self.params
        cm_t = ChaoticMap(pwlcm_map, self.seeds['t'], self.params['p_t'], self.warmup)
        D1 = (cm_q.generate(N) * 255).astype(np.uint8)
        D2 = (cm_t.generate(N) * 255).astype(np.uint8)
        # Apply two XOR layers
        flat = ch.flatten().astype(np.uint8)
        x = flat ^ D1
        x = x     ^ D2

        # 2) Confusion stage
        # 2a) dynamic DNA substitution
        R = (self.cm_log.generate(N) * 8).astype(int) % 8
        dna_plain = self.dna.encode(x, R)
        # 2b) DNA XOR with chaotic sine
        dna_key = self.dna.encode((self.cm_sin.generate(N)*255).astype(int), R)
        dna_xor = [ ''.join(DNA_XOR[(p[i],k[i])] for i in range(4))
                    for p,k in zip(dna_plain, dna_key) ]
        # 2c) fixed decode (rule 3)
        x = self.dna.decode(dna_xor, np.full(N,2))
        # --- DEBUG: check x distribution after DNA-XOR + fixed decode ---
        hist, _ = np.histogram(x, bins=10, range=(0,255))
        print(f"[DEBUG] Post-DNA decode histogram: {hist.tolist()}")
        print(f"[DEBUG] x sample[0:10]: {x[:10].tolist()}")


        # 3) Inverse diffusion to enhance mixing
        x = x ^ D2
        x = x ^ D1

        # 4) Permutation: split into 4 blocks
        img = x.reshape(H, W)
        h2, w2 = H//2, W//2
        blocks = [ img[i*h2:(i+1)*h2, j*w2:(j+1)*w2]
                   for i in range(2) for j in range(2) ]
        sel = (self.cm_sel.generate(4)*4).astype(int) % 4
        cmap_list = [quadratic_map, pwlcm_map, logistic_map, sine_map]
        out_blocks = []
        for block, idx in zip(blocks, sel):
            cm_b = ChaoticMap(cmap_list[idx], self.seeds['blk'], self.params['p'][idx], warmup=0)
            flatb = block.flatten()
            perm = build_permutation(cm_b.generate(flatb.size))
            out_blocks.append(flatb[perm].reshape(block.shape))
        # Reassemble
        top = np.hstack(out_blocks[:2])
        bot = np.hstack(out_blocks[2:])
        return np.vstack((top, bot))

    def encrypt(self, img):
        """Apply encrypt_channel to each of R,G,B"""
        return np.stack([self.encrypt_channel(img[:,:,i]) for i in range(3)], axis=2)
