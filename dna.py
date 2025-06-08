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

    def encode(self, flat_pixels, rule_indices):
        dna_list = []
        for pix, r in zip(flat_pixels, rule_indices):
            bits = f"{pix:08b}"
            rule = self.RULES[r]
            dna_list.append(''.join(rule[bits[i:i+2]] for i in range(0, 8, 2)))
        return dna_list

    def decode(self, dna_list, rule_indices):
        pixels = []
        for dna, r in zip(dna_list, rule_indices):
            inv = {v:k for k,v in self.RULES[r].items()}
            bits = ''.join(inv[n] for n in dna)
            pixels.append(int(bits, 2))
        return np.array(pixels, dtype=np.uint8)