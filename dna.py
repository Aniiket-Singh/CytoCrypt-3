# File: dna.py
import numpy as np

class DNAEncoder:
    RULES = [
        {'00':'A','01':'C','10':'G','11':'T'},
        {'00':'A','01':'G','10':'C','11':'T'},
        {'00':'C','01':'A','10':'T','11':'G'},
        {'00':'C','01':'T','10':'A','11':'G'},
        {'00':'G','01':'A','10':'T','11':'C'},
        {'00':'G','01':'T','10':'A','11':'C'},
        {'00':'T','01':'C','10':'G','11':'A'},
        {'00':'T','01':'G','10':'C','11':'A'},
    ]

    def encode(self, pixels, rules):
        dna = []
        for pix, r in zip(pixels, rules):
            bits = f"{pix:08b}"
            table = self.RULES[r]
            dna.append(''.join(table[bits[i:i+2]] for i in range(0,8,2)))
        return dna

    def decode(self, dna, rules):
        pixels = []
        for d, r in zip(dna, rules):
            inv = {v:k for k,v in self.RULES[r].items()}
            bits = ''.join(inv[ch] for ch in d)
            pixels.append(int(bits,2))
        return np.array(pixels, dtype=np.uint8)