import os
import time
import numpy as np
import cv2
import hashlib
import argparse
from scipy.integrate import solve_ivp

# DNA coding rules as defined in Table 1 of the paper
DNA_RULES = {
    1: {'A': '00', 'T': '11', 'G': '10', 'C': '01'},
    2: {'A': '00', 'T': '11', 'G': '01', 'C': '10'},
    3: {'A': '01', 'T': '10', 'G': '00', 'C': '11'},
    4: {'A': '01', 'T': '10', 'G': '11', 'C': '00'},
    5: {'A': '10', 'T': '01', 'G': '00', 'C': '11'},
    6: {'A': '10', 'T': '01', 'G': '11', 'C': '00'},
    7: {'A': '11', 'T': '00', 'G': '01', 'C': '10'},
    8: {'A': '11', 'T': '00', 'G': '10', 'C': '01'}
}

# Reverse DNA rules for decoding
DNA_DECODE_RULES = {rule: {v: k for k, v in mapping.items()} for rule, mapping in DNA_RULES.items()}

# DNA operation tables (only XOR is used)
DNA_XOR_TABLE = {
    'A': {'A': 'A', 'T': 'G', 'C': 'A', 'G': 'T'},
    'T': {'A': 'G', 'T': 'C', 'C': 'T', 'G': 'A'},
    'C': {'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'},
    'G': {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
}

class ImageEncryptor:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.original_image = None
        self.height, self.width = 0, 0
        self.encrypted_image = None
        self.decrypted_image = None
        self.row_index = 42  # Fixed permutation parameters
        self.col_index = 24
        self.A = None
        self.B = None
        
        if image_path:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Image not found or invalid format: {image_path}")
            self.height, self.width, _ = self.original_image.shape

    def load_encrypted_image(self, image_path):
        self.encrypted_image = cv2.imread(image_path)
        if self.encrypted_image is None:
            raise ValueError(f"Encrypted image not found: {image_path}")
        self.height, self.width, _ = self.encrypted_image.shape

    def _3d_permutation_vectorized(self, channel):
        """Vectorized 3D permutation"""
        h, w = channel.shape
        h_block, w_block = h // 2, w // 2
        output = np.zeros_like(channel)
        
        i, j = np.indices((h, w))
        block_i = i // h_block
        block_j = j // w_block
        local_i = i % h_block
        local_j = j % w_block
        s = local_i + local_j + self.row_index + self.col_index
        even_mask = (s % 2 == 0)
        
        new_i = np.where(
            even_mask,
            (h_block - 1 - local_i + self.row_index) % h_block,
            (local_i + self.row_index) % h_block
        )
        new_j = np.where(
            even_mask,
            (w_block - 1 - local_j + self.col_index) % w_block,
            (local_j + self.col_index) % w_block
        )
        
        new_i_global = (new_i + block_i * h_block).astype(int)
        new_j_global = (new_j + block_j * w_block).astype(int)
        output[i, j] = channel[new_i_global, new_j_global]
        
        return output
    
    def _inverse_3d_permutation_vectorized(self, channel):
        """Inverse of the 3D permutation"""
        h, w = channel.shape
        h_block, w_block = h // 2, w // 2
        output = np.zeros_like(channel)
        
        i, j = np.indices((h, w))
        block_i = i // h_block
        block_j = j // w_block
        local_i = i % h_block
        local_j = j % w_block
        s = local_i + local_j + self.row_index + self.col_index
        even_mask = (s % 2 == 0)
        
        new_i = np.where(
            even_mask,
            (h_block - 1 - local_i + self.row_index) % h_block,
            (local_i + self.row_index) % h_block
        )
        new_j = np.where(
            even_mask,
            (w_block - 1 - local_j + self.col_index) % w_block,
            (local_j + self.col_index) % w_block
        )
        
        new_i_global = (new_i + block_i * h_block).astype(int)
        new_j_global = (new_j + block_j * w_block).astype(int)
        output[new_i_global, new_j_global] = channel[i, j]
        
        return output
    
    def _generate_keys(self, image_bytes):
        """Generate keys using SHA-256"""
        sha_hash = hashlib.sha256(image_bytes).digest()
        V = np.frombuffer(sha_hash, dtype=np.uint8)
        
        K = np.zeros(8)
        for i in range(8):
            val = (V[4*i] << 24) | (V[4*i+1] << 16) | (V[4*i+2] << 8) | V[4*i+3]
            prev = 1 if i == 0 else K[i-1]
            K[i] = (prev + 5 * val) / (2**64)
        
        A_bar = np.array([4, 3, 2, 2, 0.4, 0.3])
        A = np.zeros(6)
        A[:3] = A_bar[:3] + (np.sum(K[:4]) * 2**10 * 255) % 256 / 256
        A[3] = A_bar[3] + (np.sum(K[2:6]) * 2**10 * 255) % 256 / 256
        A[4] = A_bar[4] + (np.sum(K[4:8]) * 2**10 * 255) % 256 / 256
        A[5] = A_bar[5] + (np.sum(K[4:8]) * 2**10 * 255) % 256 / 256
        
        B = np.array([A[i % 6] for i in range(8)])
        return K, A, B
    
    def _motdcm(self, initial, length, discard=500):
        """Optimized MOTDCM with precomputation"""
        alpha, beta, gamma1, gamma2 = 6, 5, 1, 1
        x, y = initial[0], initial[1]
        total_steps = discard + length
        seq_x = np.zeros(length)
        seq_y = np.zeros(length)
        
        for i in range(total_steps):
            x = (4 * alpha * x * (1 - x) + gamma1 * y**2) % 1
            y = (12 * beta * np.sin(np.pi * y) * (1 - 3 * np.sin(np.pi * y)) + gamma2 * x**2) % 1
            if i >= discard:
                idx = i - discard
                seq_x[idx] = x
                seq_y[idx] = y
                
        return seq_x, seq_y
    
    def _fhccs(self, initial, length, discard=500):
        """Optimized FHCCS with caching"""
        def equations(t, state, a, b, c):
            x1, x2, x3, x4, x5 = state
            dx1 = a * x5 * np.sin(np.pi * x2)
            dx2 = b * x1 * x4
            dx3 = c * x4
            dx4 = -c * x3 - b * x1 * x2
            dx5 = -a * x1 * np.sin(np.pi * x2)
            return [dx1, dx2, dx3, dx4, dx5]
        
        sol = solve_ivp(
            equations, 
            [0, 100], 
            initial[:5], 
            args=(10, 35, 25), 
            t_eval=np.linspace(0, 100, discard + length),
            method='RK45'
        )
        return sol.y[0, discard:], sol.y[1, discard:]
    
    def _global_scrambling(self, channel, seq):
        """Optimized global scrambling with argsort"""
        h, w = channel.shape
        flat = channel.flatten()
        flat_len = h * w
        perm_indices = np.argsort(seq[:flat_len])
        scrambled = flat[perm_indices].reshape((h, w))
        return scrambled, perm_indices
    
    def _reverse_global_scrambling(self, channel, perm_indices):
        """Reverse global scrambling"""
        h, w = channel.shape
        flat = channel.flatten()
        inv_perm = np.argsort(perm_indices)
        unscrambled = flat[inv_perm].reshape((h, w))
        return unscrambled
    
    def _1d_diffusion(self, channel, seq):
        """Improved CBC-mode diffusion"""
        arr = channel.flatten().copy()
        n = len(arr)
        terms = (seq[:n] * 255).astype(np.uint8)
        
        arr[0] = (arr[0] ^ terms[0]) & 0xFF
        for i in range(1, n):
            arr[i] = (arr[i] ^ arr[i-1] ^ terms[i]) & 0xFF
        return arr.reshape(channel.shape)
    
    def _reverse_1d_diffusion(self, channel, seq):
        """Reverse CBC-mode diffusion"""
        arr = channel.flatten().copy()
        n = len(arr)
        terms = (seq[:n] * 255).astype(np.uint8)
        
        for i in range(n-1, 0, -1):
            arr[i] = (arr[i] ^ arr[i-1] ^ terms[i]) & 0xFF
        arr[0] = (arr[0] ^ terms[0]) & 0xFF
        return arr.reshape(channel.shape)
    
    def _dna_encode(self, channel, rule):
        """Encode a channel using DNA coding rule"""
        binary = np.unpackbits(channel.astype(np.uint8).reshape(-1, 1), axis=1)[:, :8]
        bases = binary.reshape(-1, 4, 2)
        encoded = np.zeros((bases.shape[0], 4), dtype='U1')
        
        for i in range(bases.shape[0]):
            for j in range(4):
                base_bits = ''.join(str(bit) for bit in bases[i, j])
                encoded[i, j] = DNA_DECODE_RULES[rule][base_bits]
        
        return encoded.reshape(channel.shape[0], channel.shape[1], 4)
    
    def _dna_decode(self, dna_sequence, rule):
        """Decode DNA sequence to pixel values"""
        flat_seq = dna_sequence.reshape(-1, 4)
        decoded = np.zeros((flat_seq.shape[0], 8), dtype=np.uint8)
        
        for i, bases in enumerate(flat_seq):
            bin_str = ''
            for base in bases:
                bin_str += DNA_RULES[rule][base]
            decoded[i] = [int(bit) for bit in bin_str]
        
        pixels = np.packbits(decoded, axis=1, bitorder='big').flatten()
        return pixels.reshape(dna_sequence.shape[0], dna_sequence.shape[1])
    
    def _dna_operation(self, seq1, seq2):
        """Perform DNA XOR operation between two sequences"""
        result = np.zeros_like(seq1)
        for i in range(seq1.shape[0]):
            for j in range(seq1.shape[1]):
                for k in range(4):
                    base1 = seq1[i, j, k]
                    base2 = seq2[i, j, k]
                    result[i, j, k] = DNA_XOR_TABLE[base1][base2]
        return result
    
    def _process_channel_encrypt(self, channel_idx):
        """Process a single channel for encryption"""
        channel = self.original_image[:, :, channel_idx].copy()
        permuted = self._3d_permutation_vectorized(channel)
        seq_x, seq_y = self._motdcm(self.A, self.height * self.width)
        seq_s1, seq_s2 = self._fhccs(self.B, self.height * self.width)
        scrambled, _ = self._global_scrambling(permuted, seq_s1)
        diffused = self._1d_diffusion(scrambled, seq_x)
        
        encoding_rule = int(seq_s2[0] * 8) % 8 + 1
        key_rule = int(seq_s2[1] * 8) % 8 + 1
        decoding_rule = int(seq_s2[3] * 8) % 8 + 1
        
        segment = seq_y[:self.height * self.width]
        key_matrix = (segment.reshape((self.height, self.width)) * 255).astype(np.uint8)
        
        encoded_image = self._dna_encode(diffused, encoding_rule)
        encoded_key = self._dna_encode(key_matrix, key_rule)
        result_dna = self._dna_operation(encoded_image, encoded_key)
        encrypted = self._dna_decode(result_dna, decoding_rule)
        
        return encrypted
    
    def _process_channel_decrypt(self, channel_idx, encrypted_channel):
        """Process a single channel for decryption"""
        seq_x, seq_y = self._motdcm(self.A, self.height * self.width)
        seq_s1, seq_s2 = self._fhccs(self.B, self.height * self.width)
        encoding_rule = int(seq_s2[0] * 8) % 8 + 1
        key_rule = int(seq_s2[1] * 8) % 8 + 1
        decoding_rule = int(seq_s2[3] * 8) % 8 + 1
        
        segment = seq_y[:self.height * self.width]
        key_matrix = (segment.reshape((self.height, self.width)) * 255).astype(np.uint8)
        
        encoded_encrypted = self._dna_encode(encrypted_channel, decoding_rule)
        encoded_key = self._dna_encode(key_matrix, key_rule)
        result_dna = self._dna_operation(encoded_encrypted, encoded_key)
        diffused = self._dna_decode(result_dna, encoding_rule)
        
        scrambled = self._reverse_1d_diffusion(diffused, seq_x)
        _, perm_indices = self._global_scrambling(
            np.zeros((self.height, self.width)), seq_s1
        )
        permuted = self._reverse_global_scrambling(scrambled, perm_indices)
        decrypted = self._inverse_3d_permutation_vectorized(permuted)
        
        return decrypted
    
    def encrypt(self):
        """Encrypt the image"""
        if self.A is None or self.B is None:
            _, self.A, self.B = self._generate_keys(self.original_image.tobytes())
        
        channels = []
        for ch in range(3):
            channels.append(self._process_channel_encrypt(ch))
        self.encrypted_image = np.stack(channels, axis=-1)
        return self.encrypted_image
    
    def decrypt(self):
        """Decrypt the image"""
        channels = []
        for ch in range(3):
            decrypted_ch = self._process_channel_decrypt(ch, self.encrypted_image[:, :, ch])
            channels.append(decrypted_ch)
        self.decrypted_image = np.stack(channels, axis=-1)
        return self.decrypted_image
    
    def save_keys(self, key_file):
        """Save encryption keys to JSON file"""
        import json
        with open(key_file, 'w') as f:
            json.dump({
                'A': self.A.tolist(),
                'B': self.B.tolist()
            }, f)
    
    def load_keys(self, key_file):
        """Load encryption keys from JSON file"""
        import json
        with open(key_file) as f:
            keys = json.load(f)
            self.A = np.array(keys['A'])
            self.B = np.array(keys['B'])

def main():
    parser = argparse.ArgumentParser(description='Image Encryption/Decryption Tool')
    parser.add_argument('mode', choices=['enc', 'dec'], help='Operation mode: enc for encryption, dec for decryption')
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output file path (default: encrypted.png or decrypted.png)')
    parser.add_argument('-k', '--keyfile', help='Key file path (required for decryption)')
    
    args = parser.parse_args()
    
    if args.mode == 'enc':
        encryptor = ImageEncryptor(args.input)
        encrypted = encryptor.encrypt()
        output_file = args.output or 'encrypted.png'
        cv2.imwrite(output_file, encrypted)
        
        key_file = output_file + '.keys.json'
        encryptor.save_keys(key_file)
        print(f"Encryption complete. Encrypted image saved to {output_file}")
        print(f"Encryption keys saved to {key_file}")
        
    elif args.mode == 'dec':
        if not args.keyfile:
            raise ValueError("Key file is required for decryption (use -k)")
            
        encryptor = ImageEncryptor()
        encryptor.load_encrypted_image(args.input)
        encryptor.load_keys(args.keyfile)
        decrypted = encryptor.decrypt()
        
        output_file = args.output or 'decrypted.png'
        cv2.imwrite(output_file, decrypted)
        print(f"Decryption complete. Decrypted image saved to {output_file}")

if __name__ == "__main__":
    main()