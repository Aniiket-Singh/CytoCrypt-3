import numpy as np
from PIL import Image
import sys
import time
import concurrent.futures
from numba import jit
import copy
import threading  # Added for thread identification

# ----------------------------
# Diagnostic Logging
# ----------------------------
def log_with_timestamp(message):
    """Log messages with timestamp and thread ID"""
    thread_id = threading.get_ident()
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}][Thread-{thread_id}] {message}")

# ----------------------------
# Warm-up Numba functions
# ----------------------------
def warmup_numba():
    log_with_timestamp("Warming up Numba functions...")
    small_arr = np.zeros(10, dtype=np.uint8)
    logistic_map(0.1, 3.9, 10)
    sine_map(0.1, 0.9, 10)
    quadratic_map(0.1, 1.8, 10)
    pwl_map(0.1, 0.3, 10)
    singer_map(0.1, 1.07, 10)
    scramble_block_numba(small_arr, np.random.rand(10))
    unscramble_block_numba(small_arr, np.random.rand(10))
    log_with_timestamp("Numba warm-up complete")

# ----------------------------
# Optimized Chaotic Maps
# ----------------------------
@jit(nopython=True)
def logistic_map(x0, r, length):
    seq = np.empty(length)
    x = x0
    for i in range(length):
        x = r * x * (1 - x)
        seq[i] = x
    return seq

@jit(nopython=True)
def sine_map(x0, a, length):
    seq = np.empty(length)
    x = x0
    for i in range(length):
        x = a * np.sin(np.pi * x)
        seq[i] = x
    return seq

@jit(nopython=True)
def quadratic_map(x0, c, length):
    seq = np.empty(length)
    x = x0
    for i in range(length):
        x = c - x**2
        seq[i] = x
    return seq

@jit(nopython=True)
def pwl_map(x0, p, length):
    seq = np.empty(length)
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

@jit(nopython=True)
def singer_map(x0, mu, length):
    seq = np.empty(length)
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
# Enhanced Cryptographic Functions
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
        
        # Enhanced diffusion parameters
        self.diff_x0 = 0.12345678
        self.diff_r = 3.999

def generate_dna_mask(height, width, x0, r):
    """Generate DNA mask using logistic map"""
    size = height * width * 4
    chaotic_seq = logistic_map(x0, r, size)
    int_mask = (chaotic_seq * 255).astype(np.uint8)
    
    mask = np.empty((height, width), dtype='U4')
    for i in range(height):
        for j in range(width):
            idx = (i * width + j) * 4
            bases = ''.join(
                dna_encode_pixel(int_mask[idx + k], 0)
                for k in range(4)
            )
            mask[i,j] = bases[:4]
    return mask

@jit(nopython=True)
def scramble_block_numba(block, seq):
    flat_block = block.flatten()
    perm_indices = np.argsort(seq)
    return flat_block[perm_indices].reshape(block.shape)

@jit(nopython=True)
def unscramble_block_numba(block, seq):
    flat_block = block.flatten()
    perm_indices = np.argsort(seq)
    inv_perm = np.argsort(perm_indices)
    return flat_block[inv_perm].reshape(block.shape)

def enhanced_dna_xor(I2_dna, D1_dna, D2_dna):
    temp_dna = dna_xor_matrix(I2_dna, D1_dna)
    feedback = np.roll(temp_dna, (1, 1), axis=(0, 1))
    temp_dna = dna_xor_matrix(temp_dna, feedback)
    return dna_xor_matrix(temp_dna, D2_dna)

def pixel_diffusion(matrix, keys):
    height, width = matrix.shape
    size = height * width
    diff_map = logistic_map(keys.diff_x0, keys.diff_r, size)
    diff_map = (diff_map * 255).astype(np.uint8).reshape(height, width)
    return np.bitwise_xor(matrix, diff_map)

# ----------------------------
# Parallel Processing Functions
# ----------------------------
def process_blocks(blocks, keys, encrypt=True):
    sequences = []
    for block in blocks:
        size = block.size
        seq = singer_map(keys.singer_x0, keys.singer_mu, size)
        sequences.append(seq)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, block in enumerate(blocks):
            if encrypt:
                futures.append(executor.submit(scramble_block_numba, block, sequences[i]))
            else:
                futures.append(executor.submit(unscramble_block_numba, block, sequences[i]))
        
        return [f.result() for f in futures]

def process_channel(channel, keys, encrypt=True):
    m, n = channel.shape
    m_pad = (2 - m % 2) % 2
    n_pad = (2 - n % 2) % 2
    padded = np.pad(channel, ((0, m_pad), (0, n_pad)), 'reflect')
    M, N = padded.shape
    total_pixels = M * N
    
    if encrypt:
        # Step 2: Dynamic DNA encoding
        rule_seq = logistic_map(keys.rule_x0, keys.rule_r, total_pixels)
        rule_indices = (rule_seq * 8).astype(int) % 8
        
        I2_dna = np.empty((M, N), dtype='U4')
        for i in range(M):
            for j in range(N):
                I2_dna[i,j] = dna_encode_pixel(padded[i,j], rule_indices[i*N + j])
        
        # Step 3: Enhanced DNA-XOR with avalanche effect
        D1_dna = generate_dna_mask(M, N, keys.d1_x0, keys.d1_r)
        D2_dna = generate_dna_mask(M, N, keys.d2_x0, keys.d2_r)
        I3_dna = enhanced_dna_xor(I2_dna, D1_dna, D2_dna)
        
        # Step 4: Convert to indices and reshape for blocks
        char_array = np.array([list(dna_str) for dna_str in I3_dna.flatten()])
        I4_int = np.vectorize({'A':0, 'T':1, 'C':2, 'G':3}.get)(char_array)
        I4_reshaped = I4_int.reshape(2*M, 2*N)
        
        # Step 5: Block scrambling (parallel)
        blocks = [
            I4_reshaped[:M, :N],
            I4_reshaped[:M, N:],
            I4_reshaped[M:, :N],
            I4_reshaped[M:, N:]
        ]
        scrambled_blocks = process_blocks(blocks, keys, encrypt=True)
        
        # Reassemble
        top = np.hstack([scrambled_blocks[0], scrambled_blocks[1]])
        bottom = np.hstack([scrambled_blocks[2], scrambled_blocks[3]])
        I5_int = np.vstack([top, bottom])
        
        # Step 6: Convert to DNA bases and repeat Step 3
        base_list = []
        for base_idx in I5_int.flatten():
            base_list.append({0: 'A', 1: 'T', 2: 'C', 3: 'G'}[base_idx])
        
        # Group every 4 bases into DNA strings
        dna_strings = []
        for i in range(0, len(base_list), 4):
            dna_strings.append(''.join(base_list[i:i+4]))
        
        I5_dna = np.array(dna_strings).reshape(M, N)
        
        # Repeat enhanced DNA-XOR
        temp_dna = enhanced_dna_xor(I5_dna, D1_dna, D2_dna)
        
        # Step 7: Decode to image
        I7 = np.zeros((M, N), dtype=np.uint8)
        for i in range(M):
            for j in range(N):
                I7[i,j] = dna_decode_pixel(temp_dna[i,j], 0)  # Rule 0 (Rule 1)
        
        # Remove padding
        result = I7[:m, :n]
        
        # Add pixel-level diffusion
        return pixel_diffusion(result, keys)
    
    else:  # Decryption
        # Reverse pixel-level diffusion
        padded = pixel_diffusion(padded, keys)
        
        # Reverse Step 7: Encode with fixed rule
        I7_dna = np.empty((M, N), dtype='U4')
        for i in range(M):
            for j in range(N):
                I7_dna[i,j] = dna_encode_pixel(padded[i,j], 0)  # Rule 0 (Rule 1)
        
        # Reverse Step 6: Double DNA-XOR
        D1_dna = generate_dna_mask(M, N, keys.d1_x0, keys.d1_r)
        D2_dna = generate_dna_mask(M, N, keys.d2_x0, keys.d2_r)
        
        temp_dna = dna_xor_matrix(I7_dna, D2_dna)
        I6_dna = dna_xor_matrix(temp_dna, D1_dna)
        
        # Prepare for block unscrambling
        char_list = []
        for dna_str in I6_dna.flatten():
            char_list.extend(list(dna_str))
        
        I5_int = np.vectorize({'A':0, 'T':1, 'C':2, 'G':3}.get)(np.array(char_list))
        I5_int = I5_int.reshape(2*M, 2*N)
        
        # Reverse Step 5: Block unscrambling (parallel)
        blocks = [
            I5_int[:M, :N],
            I5_int[:M, N:],
            I5_int[M:, :N],
            I5_int[M:, N:]
        ]
        unscrambled_blocks = process_blocks(blocks, keys, encrypt=False)
        
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
        rule_seq = logistic_map(keys.rule_x0, keys.rule_r, M*N)
        rule_indices = (rule_seq * 8).astype(int) % 8
        
        decrypted = np.zeros((M, N), dtype=np.uint8)
        for i in range(M):
            for j in range(N):
                decrypted[i,j] = dna_decode_pixel(I2_dna[i,j], rule_indices[i*N + j])
        
        # Remove padding
        decrypted = decrypted[:m, :n]
        return pixel_diffusion(decrypted, keys)  # Diffusion LAST

# ----------------------------
# Fixed Multi-round Processing
# ----------------------------
def multi_round_process_channel(channel, keys, encrypt=True, rounds=2):
    result = channel.copy()
    
    # Create key modifications for each round
    round_keys = []
    for i in range(rounds):
        rk = copy.deepcopy(keys)
        if i > 0:
            rk.rule_x0 = (keys.rule_x0 + 0.01 * i) % 1.0
            rk.d1_x0 = (keys.d1_x0 + 0.01 * i) % 1.0
            rk.d2_x0 = (keys.d2_x0 + 0.01 * i) % 1.0
            rk.singer_x0 = (keys.singer_x0 + 0.01 * i) % 1.0
            rk.diff_x0 = (keys.diff_x0 + 0.01 * i) % 1.0
        round_keys.append(rk)
    
    if not encrypt:
        round_keys.reverse()  # Reverse key order for decryption
    
    for rk in round_keys:
        result = process_channel(result, rk, encrypt)
    
    return result

# ----------------------------
# Channel Processing with Logging
# ----------------------------
def process_image_channel(channel, keys, encrypt=True, channel_name="Unknown"):
    """Process a single channel with diagnostic logging"""
    start_time = time.time()
    log_with_timestamp(f"START processing {channel_name} channel ({'encrypt' if encrypt else 'decrypt'})")
    
    result = multi_round_process_channel(channel, keys, encrypt)
    
    duration = time.time() - start_time
    log_with_timestamp(f"END processing {channel_name} channel ({'encrypt' if encrypt else 'decrypt'}) - took {duration:.2f}s")
    return result

# ----------------------------
# Main Function
# ----------------------------
def main(image_path):
    total_start = time.time()
    keys = ChaosKeys()
    
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    r, g, b = img.split()
    r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)
    
    # Encrypt
    encrypt_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        log_with_timestamp("Submitting channel encryption tasks")
        
        future_r = executor.submit(process_image_channel, r_arr, keys, True, "RED")
        future_g = executor.submit(process_image_channel, g_arr, keys, True, "GREEN")
        future_b = executor.submit(process_image_channel, b_arr, keys, True, "BLUE")
        
        log_with_timestamp("Waiting for encryption results")
        r_enc = future_r.result()
        g_enc = future_g.result()
        b_enc = future_b.result()
    
    encrypt_time = time.time() - encrypt_start
    
    # Save encrypted
    Image.merge('RGB', (
        Image.fromarray(r_enc),
        Image.fromarray(g_enc),
        Image.fromarray(b_enc)
    )).save('encrypted.png')
    
    # Decrypt
    decrypt_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        log_with_timestamp("Submitting channel decryption tasks")
        
        future_r = executor.submit(process_image_channel, r_enc, keys, False, "RED")
        future_g = executor.submit(process_image_channel, g_enc, keys, False, "GREEN")
        future_b = executor.submit(process_image_channel, b_enc, keys, False, "BLUE")
        
        log_with_timestamp("Waiting for decryption results")
        r_dec = future_r.result()
        g_dec = future_g.result()
        b_dec = future_b.result()
    
    decrypt_time = time.time() - decrypt_start
    
    # Save decrypted
    Image.merge('RGB', (
        Image.fromarray(r_dec),
        Image.fromarray(g_dec),
        Image.fromarray(b_dec)
    )).save('decrypted.png')
    
    total_time = time.time() - total_start
    log_with_timestamp(f"\nProcessing Complete!\n"
          f"Encryption: {encrypt_time:.2f}s\n"
          f"Decryption: {decrypt_time:.2f}s\n"
          f"Total: {total_time:.2f}s")

if __name__ == "__main__":
    warmup_numba()  # Pre-compile Numba functions
    if len(sys.argv) != 2:
        print("Usage: python dna_encryption.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])