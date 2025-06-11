import numpy as np
from PIL import Image
import sys
import os

# ----------------------------
# Chaotic Map Implementations
# ----------------------------
def logistic_map(x0, r, length):
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = r * x * (1 - x)
        seq[i] = x
    return seq

def sine_map(x0, a, length):
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = a * np.sin(np.pi * x)
        seq[i] = x
    return seq

def quadratic_map(x0, c, length):
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = c - x**2
        seq[i] = x
    return seq

def pwl_map(x0, p, length):
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        if x < p:
            x = x / p
        elif x < 0.5:
            x = (x - p) / (0.5 - p)
        elif x < 1 - p:
            x = (1 - p - x) / (0.5 - p)
        else:
            x = (1 - x) / p
        seq[i] = x
    return seq

def singer_map(x0, mu, length):
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = mu * (7.86*x - 23.31*x**2 + 28.75*x**3 - 13.302875*x**4)
        seq[i] = x
    return seq

# ----------------------------
# DNA Operations
# ----------------------------
# DNA encoding rules (8 variations)
DNA_ENCODE_RULES = [
    {'00': 'A', '01': 'T', '10': 'C', '11': 'G'},  # Rule 1
    {'00': 'A', '01': 'G', '10': 'C', '11': 'T'},  # Rule 2
    {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},  # Rule 3
    {'00': 'C', '01': 'T', '10': 'A', '11': 'G'},  # Rule 4
    {'00': 'G', '01': 'A', '10': 'T', '11': 'C'},  # Rule 5
    {'00': 'G', '01': 'C', '10': 'A', '11': 'T'},  # Rule 6
    {'00': 'T', '01': 'A', '10': 'G', '11': 'C'},  # Rule 7
    {'00': 'T', '01': 'G', '10': 'C', '11': 'A'}   # Rule 8
]

# DNA decoding rules (inverse mappings)
DNA_DECODE_RULES = [
    {v: k for k, v in rule.items()} for rule in DNA_ENCODE_RULES
]

# DNA XOR table
DNA_XOR_TABLE = {
    ('A','A'): 'A', ('A','T'): 'T', ('A','C'): 'C', ('A','G'): 'G',
    ('T','A'): 'T', ('T','T'): 'A', ('T','C'): 'G', ('T','G'): 'C',
    ('C','A'): 'C', ('C','T'): 'G', ('C','C'): 'A', ('C','G'): 'T',
    ('G','A'): 'G', ('G','T'): 'C', ('G','C'): 'T', ('G','G'): 'A'
}

def dna_encode_pixel(pixel, rule_idx):
    """Encode single pixel (0-255) to DNA bases using specified rule"""
    binary = f"{pixel:08b}"
    rule = DNA_ENCODE_RULES[rule_idx]
    return ''.join(rule[binary[i:i+2]] for i in range(0, 8, 2))

def dna_decode_pixel(dna_str, rule_idx):
    """Decode DNA bases to pixel (0-255) using specified rule"""
    rule = DNA_DECODE_RULES[rule_idx]
    binary = ''.join(rule[base] for base in dna_str)
    return int(binary, 2)

def dna_xor_matrix(matrix1, matrix2):
    """Element-wise DNA-XOR for two DNA base matrices"""
    result = np.empty_like(matrix1)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i,j] = ''.join(
                DNA_XOR_TABLE[(a,b)] 
                for a,b in zip(matrix1[i,j], matrix2[i,j])
            )
    return result

def bases_to_indices(dna_matrix):
    """Convert DNA base matrix to integer indices (A=0,T=1,C=2,G=3)"""
    base_map = {'A':0, 'T':1, 'C':2, 'G':3}
    return np.vectorize(base_map.get)(dna_matrix)

def indices_to_bases(index_matrix):
    """Convert index matrix (0-3) to DNA bases"""
    index_map = {0:'A', 1:'T', 2:'C', 3:'G'}
    return np.vectorize(index_map.get)(index_matrix)

# ----------------------------
# Core Cryptographic Functions
# ----------------------------
class ChaosKeys:
    """Container for chaotic system parameters"""
    def __init__(self):
        # Dynamic rule selection (Step 2)
        self.rule_x0 = 0.23456789
        self.rule_r = 3.999
        
        # DNA masks (Step 3)
        self.d1_x0 = 0.34567891
        self.d1_r = 3.95
        self.d2_x0 = 0.45678912
        self.d2_r = 3.92
        
        # Block scrambling (Step 5)
        self.singer_x0 = 0.56789123
        self.singer_mu = 1.07
        self.quadratic_x0 = 0.67891234
        self.quadratic_c = 1.8
        self.logistic_x0 = 0.78912345
        self.logistic_r = 3.99
        self.sine_x0 = 0.89123456
        self.sine_a = 0.99
        self.pwl_x0 = 0.91234567
        self.pwl_p = 0.3

def generate_dna_mask(height, width, x0, r):
    """Generate DNA mask using logistic map"""
    size = height * width * 4  # 4 bases per pixel
    chaotic_seq = logistic_map(x0, r, size)
    int_mask = (chaotic_seq * 255).astype(np.uint8)
    
    # Convert to DNA bases using fixed rule (Rule 1)
    mask = np.empty((height, width), dtype='U4')
    for i in range(height):
        for j in range(width):
            idx = (i * width + j) * 4
            bases = ''.join(
                dna_encode_pixel(int_mask[idx + k], 0)
                for k in range(4)
            )
            mask[i,j] = bases[:4]  # Take first 4 bases
    return mask

def scramble_block(block, chaotic_func, init, param):
    """Scramble block using chaotic map"""
    flat_block = block.flatten()
    length = len(flat_block)
    seq = chaotic_func(init, param, length)
    perm_indices = np.argsort(seq)
    return flat_block[perm_indices].reshape(block.shape)

def unscramble_block(block, chaotic_func, init, param):
    """Unscramble block using chaotic map"""
    flat_block = block.flatten()
    length = len(flat_block)
    seq = chaotic_func(init, param, length)
    perm_indices = np.argsort(seq)
    inv_perm = np.argsort(perm_indices)
    return flat_block[inv_perm].reshape(block.shape)

# ----------------------------
# Main Cryptographic Processes
# ----------------------------
# ====== FIXED ENCRYPTION FUNCTION ======
def encrypt_image(image_path, keys):
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    I1 = np.array(img)
    m, n = I1.shape
    
    # Pad to even dimensions
    m_pad = (2 - m % 2) % 2
    n_pad = (2 - n % 2) % 2
    I1 = np.pad(I1, ((0, m_pad), (0, n_pad)), 'reflect')
    M, N = I1.shape
    total_pixels = M * N
    
    # Step 2: Dynamic DNA encoding
    rule_seq = logistic_map(keys.rule_x0, keys.rule_r, total_pixels)
    rule_indices = (rule_seq * 8).astype(int) % 8
    
    I2_dna = np.empty((M, N), dtype='U4')
    for i in range(M):
        for j in range(N):
            I2_dna[i,j] = dna_encode_pixel(I1[i,j], rule_indices[i*N + j])
    
    # Step 3: Double DNA-XOR
    D1_dna = generate_dna_mask(M, N, keys.d1_x0, keys.d1_r)
    D2_dna = generate_dna_mask(M, N, keys.d2_x0, keys.d2_r)
    
    temp_dna = dna_xor_matrix(I2_dna, D1_dna)
    I3_dna = dna_xor_matrix(temp_dna, D2_dna)
    
    # Step 4: Convert to indices and reshape for blocks
    char_array = np.array([list(dna_str) for dna_str in I3_dna.flatten()])
    I4_int = np.vectorize({'A':0, 'T':1, 'C':2, 'G':3}.get)(char_array)
    I4_reshaped = I4_int.reshape(2*M, 2*N)
    
    # Step 5: Block scrambling
    blocks = [
        I4_reshaped[:M, :N],
        I4_reshaped[:M, N:],
        I4_reshaped[M:, :N],
        I4_reshaped[M:, N:]
    ]
    
    # Assign chaotic maps to blocks
    singer_seq = singer_map(keys.singer_x0, keys.singer_mu, 4)
    map_assign = (singer_seq * 4).astype(int) % 4
    
    chaotic_funcs = [
        lambda x, p, l: quadratic_map(x, p, l),
        lambda x, p, l: logistic_map(x, p, l),
        lambda x, p, l: sine_map(x, p, l),
        lambda x, p, l: pwl_map(x, p, l)
    ]
    
    params = [
        keys.quadratic_c,
        keys.logistic_r,
        keys.sine_a,
        keys.pwl_p
    ]
    
    inits = [
        keys.quadratic_x0,
        keys.logistic_x0,
        keys.sine_x0,
        keys.pwl_x0
    ]
    
    # Scramble blocks
    scrambled_blocks = []
    for i in range(4):
        func_idx = map_assign[i]
        scrambled = scramble_block(
            blocks[i],
            chaotic_funcs[func_idx],
            inits[func_idx],
            params[func_idx]
        )
        scrambled_blocks.append(scrambled)
    
    # Reassemble
    top = np.hstack([scrambled_blocks[0], scrambled_blocks[1]])
    bottom = np.hstack([scrambled_blocks[2], scrambled_blocks[3]])
    I5_int = np.vstack([top, bottom])
    
    # Step 6: Convert to DNA bases and repeat Step 3
    # FIX: Properly handle DNA string formation
    base_list = []
    for base_idx in I5_int.flatten():
        base_list.append({0: 'A', 1: 'T', 2: 'C', 3: 'G'}[base_idx])
    
    # Group every 4 bases into DNA strings
    dna_strings = []
    for i in range(0, len(base_list), 4):
        dna_strings.append(''.join(base_list[i:i+4]))
    
    I5_dna = np.array(dna_strings).reshape(M, N)
    
    # Repeat Step 3 DNA-XOR
    temp_dna = dna_xor_matrix(I5_dna, D1_dna)
    I6_dna = dna_xor_matrix(temp_dna, D2_dna)
    
    # Step 7: Decode to image
    I7 = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            I7[i,j] = dna_decode_pixel(I6_dna[i,j], 0)  # Rule 0 (Rule 1)
    
    # Remove padding
    encrypted = I7[:m, :n]
    return encrypted

# ====== FIXED DECRYPTION FUNCTION ======
def decrypt_image(encrypted, keys):
    # Pad encrypted image
    m, n = encrypted.shape
    m_pad = (2 - m % 2) % 2
    n_pad = (2 - n % 2) % 2
    I7 = np.pad(encrypted, ((0, m_pad), (0, n_pad)), 'reflect')
    M, N = I7.shape
    total_pixels = M * N
    
    # Reverse Step 7: Encode with fixed rule
    I7_dna = np.empty((M, N), dtype='U4')
    for i in range(M):
        for j in range(N):
            I7_dna[i,j] = dna_encode_pixel(I7[i,j], 0)  # Rule 0 (Rule 1)
    
    # Reverse Step 6: Double DNA-XOR
    D1_dna = generate_dna_mask(M, N, keys.d1_x0, keys.d1_r)
    D2_dna = generate_dna_mask(M, N, keys.d2_x0, keys.d2_r)
    
    temp_dna = dna_xor_matrix(I7_dna, D2_dna)
    I6_dna = dna_xor_matrix(temp_dna, D1_dna)
    
    # FIX: Prepare for block unscrambling
    char_list = []
    for dna_str in I6_dna.flatten():
        char_list.extend(list(dna_str))
    
    I5_int = np.vectorize({'A':0, 'T':1, 'C':2, 'G':3}.get)(np.array(char_list))
    I5_int = I5_int.reshape(2*M, 2*N)
    
    # Reverse Step 5: Block unscrambling
    blocks = [
        I5_int[:M, :N],
        I5_int[:M, N:],
        I5_int[M:, :N],
        I5_int[M:, N:]
    ]
    
    # Same map assignment
    singer_seq = singer_map(keys.singer_x0, keys.singer_mu, 4)
    map_assign = (singer_seq * 4).astype(int) % 4
    
    chaotic_funcs = [
        lambda x, p, l: quadratic_map(x, p, l),
        lambda x, p, l: logistic_map(x, p, l),
        lambda x, p, l: sine_map(x, p, l),
        lambda x, p, l: pwl_map(x, p, l)
    ]
    
    params = [
        keys.quadratic_c,
        keys.logistic_r,
        keys.sine_a,
        keys.pwl_p
    ]
    
    inits = [
        keys.quadratic_x0,
        keys.logistic_x0,
        keys.sine_x0,
        keys.pwl_x0
    ]
    
    unscrambled_blocks = []
    for i in range(4):
        func_idx = map_assign[i]
        unscrambled = unscramble_block(
            blocks[i],
            chaotic_funcs[func_idx],
            inits[func_idx],
            params[func_idx]
        )
        unscrambled_blocks.append(unscrambled)
    
    # Reassemble
    top = np.hstack([unscrambled_blocks[0], unscrambled_blocks[1]])
    bottom = np.hstack([unscrambled_blocks[2], unscrambled_blocks[3]])
    I4_int = np.vstack([top, bottom])
    
    # Reverse Step 4: Convert to DNA bases
    base_list = []
    for base_idx in I4_int.flatten():
        base_list.append({0: 'A', 1: 'T', 2: 'C', 3: 'G'}[base_idx])
    
    # Group every 4 bases into DNA strings
    dna_strings = []
    for i in range(0, len(base_list), 4):
        dna_strings.append(''.join(base_list[i:i+4]))
    
    I3_dna = np.array(dna_strings).reshape(M, N)
    
    # Reverse Step 3: Double DNA-XOR
    temp_dna = dna_xor_matrix(I3_dna, D2_dna)
    I2_dna = dna_xor_matrix(temp_dna, D1_dna)
    
    # Reverse Step 2: Dynamic DNA decoding
    rule_seq = logistic_map(keys.rule_x0, keys.rule_r, total_pixels)
    rule_indices = (rule_seq * 8).astype(int) % 8
    
    decrypted = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            decrypted[i,j] = dna_decode_pixel(I2_dna[i,j], rule_indices[i*N + j])
    
    # Remove padding
    return decrypted[:m, :n]

# ----------------------------
# CLI Interface
# ----------------------------
def main(image_path):
    # Initialize keys (in real use, save/load these)
    keys = ChaosKeys()
    
    # Encrypt
    encrypted = encrypt_image(image_path, keys)
    Image.fromarray(encrypted).save('encrypted.png')
    
    # Decrypt
    decrypted = decrypt_image(encrypted, keys)
    Image.fromarray(decrypted).save('decrypted.png')
    
    print("Encryption/decryption complete")
    print(f"Encrypted: encrypted.png")
    print(f"Decrypted: decrypted.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dna_encryption.py <image_path>")
        sys.exit(1)
    
    main(sys.argv[1])