import os
import time
import numpy as np
import cv2
import hashlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.util import random_noise
import math
import json
from tqdm import tqdm
from hilbertcurve.hilbertcurve import HilbertCurve

os.makedirs('results', exist_ok=True)

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

# DNA operation tables
DNA_ADD_TABLE = {
    'A': {'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'},
    'T': {'A': 'T', 'T': 'C', 'C': 'G', 'G': 'A'},
    'C': {'A': 'C', 'T': 'G', 'C': 'A', 'G': 'T'},
    'G': {'A': 'G', 'T': 'A', 'C': 'T', 'G': 'C'}
}

DNA_SUB_TABLE = {
    'A': {'A': 'A', 'T': 'G', 'C': 'T', 'G': 'C'},
    'T': {'A': 'C', 'T': 'A', 'C': 'G', 'G': 'T'},
    'C': {'A': 'T', 'T': 'C', 'C': 'A', 'G': 'G'},
    'G': {'A': 'G', 'T': 'T', 'C': 'C', 'G': 'A'}
}

DNA_XOR_TABLE = {
    'A': {'A': 'A', 'T': 'G', 'C': 'A', 'G': 'T'},
    'T': {'A': 'G', 'T': 'C', 'C': 'T', 'G': 'A'},
    'C': {'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'},
    'G': {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
}

DNA_XNOR_TABLE = {
    'A': {'A': 'A', 'T': 'T', 'C': 'C', 'G': 'G'},
    'T': {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'},
    'C': {'A': 'C', 'T': 'G', 'C': 'A', 'G': 'T'},
    'G': {'A': 'G', 'T': 'C', 'C': 'T', 'G': 'A'}
}

DNA_OPERATIONS = {
    1: DNA_ADD_TABLE,
    2: DNA_SUB_TABLE,
    3: DNA_XOR_TABLE,
    4: DNA_XNOR_TABLE
}

class ImageEncryptor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Image not found or invalid format: {image_path}")
        self.height, self.width, _ = self.original_image.shape
        self.encrypted_image = None
        self.decrypted_image = None
        self.row_index = 42  # Arbitrary user input
        self.col_index = 24   # Arbitrary user input
        self.results = {}
        self.original_image_backup = self.original_image.copy()
        self.A = None
        self.B = None
        self.encryption_sequences = {}
        self.encryption_time = 0
        self.decryption_time = 0
        self.key_gen_time = 0
        self.dna_time = 0
        self.dna_operations = {}
        
    def _3d_permutation_vectorized(self, channel):
        """Vectorized 3D permutation"""
        h, w = channel.shape
        h_block, w_block = h // 2, w // 2
        output = np.zeros_like(channel)
        
        # Create coordinate grids
        i, j = np.indices((h, w))
        
        # Calculate block indices
        block_i = i // h_block
        block_j = j // w_block
        
        # Calculate local coordinates within blocks
        local_i = i % h_block
        local_j = j % w_block
        
        # Calculate sum condition
        s = local_i + local_j + self.row_index + self.col_index
        even_mask = (s % 2 == 0)
        
        # Calculate new positions
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
        
        # Convert to global coordinates AND CAST TO INT
        new_i_global = (new_i + block_i * h_block).astype(int)
        new_j_global = (new_j + block_j * w_block).astype(int)
        
        # Apply permutation
        output[i, j] = channel[new_i_global, new_j_global]
        
        return output
    
    def _inverse_3d_permutation_vectorized(self, channel):
        """Inverse of the 3D permutation"""
        h, w = channel.shape
        h_block, w_block = h // 2, w // 2
        output = np.zeros_like(channel)
        
        # Create coordinate grids
        i, j = np.indices((h, w))
        
        # Calculate block indices
        block_i = i // h_block
        block_j = j // w_block
        
        # Calculate local coordinates within blocks
        local_i = i % h_block
        local_j = j % w_block
        
        # Calculate sum condition
        s = local_i + local_j + self.row_index + self.col_index
        even_mask = (s % 2 == 0)
        
        # Calculate new positions (same as forward)
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
        
        # Convert to global coordinates AND CAST TO INT
        new_i_global = (new_i + block_i * h_block).astype(int)
        new_j_global = (new_j + block_j * w_block).astype(int)
        
        # Apply inverse permutation
        output[new_i_global, new_j_global] = channel[i, j]
        
        return output
    
    def _generate_keys(self, image_bytes):
        """Generate keys using SHA-256"""
        start_time = time.time()
        
        # Compute SHA-256 hash
        sha_hash = hashlib.sha256(image_bytes).digest()
        V = np.frombuffer(sha_hash, dtype=np.uint8)
        
        # Vectorized key generation
        K = np.zeros(8)
        for i in range(8):
            val = (V[4*i] << 24) | (V[4*i+1] << 16) | (V[4*i+2] << 8) | V[4*i+3]
            prev = 1 if i == 0 else K[i-1]
            K[i] = (prev + 5 * val) / (2**64)
        
        # MOTDCM initial values
        A_bar = np.array([4, 3, 2, 2, 0.4, 0.3])
        A = np.zeros(6)
        A[:3] = A_bar[:3] + (np.sum(K[:4]) * 2**10 * 255) % 256 / 256
        A[3] = A_bar[3] + (np.sum(K[2:6]) * 2**10 * 255) % 256 / 256
        A[4] = A_bar[4] + (np.sum(K[4:8]) * 2**10 * 255) % 256 / 256
        A[5] = A_bar[5] + (np.sum(K[4:8]) * 2**10 * 255) % 256 / 256
        
        # FHCCS initial values
        B = np.array([A[i % 6] for i in range(8)])
        
        self.key_gen_time = time.time() - start_time
        return K, A, B
    
    def _motdcm(self, initial, length, discard=500):
        """Optimized MOTDCM with precomputation"""
        alpha, beta, gamma1, gamma2 = 6, 5, 1, 1
        x, y = initial[0], initial[1]
        total_steps = discard + length
        seq_x = np.zeros(length)
        seq_y = np.zeros(length)
        
        # Precompute values
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
        # Solve ODE system
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
        
        # Return last two states
        return sol.y[0, discard:], sol.y[1, discard:]
    
    def _global_scrambling(self, channel, seq):
        """Optimized global scrambling with argsort"""
        h, w = channel.shape
        flat = channel.flatten()
        flat_len = h * w
        
        # Use argsort directly on the sequence
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
        
        # Initialize with first pixel
        arr[0] = (arr[0] ^ terms[0]) & 0xFF
        
        # Propagate changes through the image
        for i in range(1, n):
            arr[i] = (arr[i] ^ arr[i-1] ^ terms[i]) & 0xFF
        
        return arr.reshape(channel.shape)
    
    def _reverse_1d_diffusion(self, channel, seq):
        """Reverse CBC-mode diffusion"""
        arr = channel.flatten().copy()
        n = len(arr)
        terms = (seq[:n] * 255).astype(np.uint8)
        
        # Process in reverse order
        for i in range(n-1, 0, -1):
            arr[i] = (arr[i] ^ arr[i-1] ^ terms[i]) & 0xFF
        
        # Handle first pixel
        arr[0] = (arr[0] ^ terms[0]) & 0xFF
        
        return arr.reshape(channel.shape)
    
    def _dna_encode(self, channel, rule):
        """Encode a channel using DNA coding rule"""
        start_time = time.time()
        
        # Convert to binary strings
        binary = np.unpackbits(channel.astype(np.uint8).reshape(-1, 1), axis=1)[:, :8]
        
        # Reshape to 4 bases per pixel (each base = 2 bits)
        bases = binary.reshape(-1, 4, 2)
        
        # Convert to DNA sequence
        encoded = np.zeros((bases.shape[0], 4), dtype='U1')
        for i in range(bases.shape[0]):
            for j in range(4):
                base_bits = ''.join(str(bit) for bit in bases[i, j])
                # Ensure rule exists in decode rules
                if rule not in DNA_DECODE_RULES:
                    raise ValueError(f"Invalid DNA rule: {rule}")
                if base_bits not in DNA_DECODE_RULES[rule]:
                    raise ValueError(f"Invalid binary sequence: {base_bits} for rule {rule}")
                encoded[i, j] = DNA_DECODE_RULES[rule][base_bits]
        
        self.dna_time += time.time() - start_time
        return encoded.reshape(channel.shape[0], channel.shape[1], 4)
    
    def _dna_decode(self, dna_sequence, rule):
        """Decode DNA sequence to pixel values"""
        start_time = time.time()
        
        # Flatten DNA sequence
        flat_seq = dna_sequence.reshape(-1, 4)
        decoded = np.zeros((flat_seq.shape[0], 8), dtype=np.uint8)
        
        # Convert DNA to binary
        for i, bases in enumerate(flat_seq):
            bin_str = ''
            for base in bases:
                # Ensure rule exists in DNA rules
                if rule not in DNA_RULES:
                    raise ValueError(f"Invalid DNA rule: {rule}")
                if base not in DNA_RULES[rule]:
                    raise ValueError(f"Invalid DNA base: {base} for rule {rule}")
                bin_str += DNA_RULES[rule][base]
            decoded[i] = [int(bit) for bit in bin_str]
        
        # Convert binary to uint8 - FIXED: use big-endian packing
        pixels = np.packbits(decoded, axis=1, bitorder='big').flatten()
        
        self.dna_time += time.time() - start_time
        return pixels.reshape(dna_sequence.shape[0], dna_sequence.shape[1])
    
    def _dna_operation(self, seq1, seq2, op_rule):
        """Perform DNA operation between two sequences - XOR ONLY"""
        start_time = time.time()
        
        # Only allow XOR (operation_rule=3) as it's self-inverse
        if op_rule != 3:
            print(f"WARNING: Using XOR instead of operation {op_rule}")
        op_table = DNA_XOR_TABLE
        
        result = np.zeros_like(seq1)
        for i in range(seq1.shape[0]):
            for j in range(seq1.shape[1]):
                for k in range(4):
                    base1 = seq1[i, j, k]
                    base2 = seq2[i, j, k]
                    result[i, j, k] = op_table[base1][base2]
        
        self.dna_time += time.time() - start_time
        return result
    
    def _process_channel_encrypt(self, channel_idx):
        """Process a single channel for encryption"""
        channel = self.original_image[:, :, channel_idx].copy()
        
        # 1. 3D Permutation
        permuted = self._3d_permutation_vectorized(channel)
        
        # 2. Generate chaotic sequences
        seq_x, seq_y = self._motdcm(self.A, self.height * self.width)
        seq_s1, seq_s2 = self._fhccs(self.B, self.height * self.width)
        
        # 3. Global scrambling
        scrambled, perm_indices = self._global_scrambling(permuted, seq_s1)
        
        # 4. 1D Diffusion
        diffused = self._1d_diffusion(scrambled, seq_x)
        
        # 5. DNA operations - FORCE XOR OPERATION
        # Determine DNA rules from chaotic sequence
        encoding_rule = int(seq_s2[0] * 8) % 8 + 1
        key_rule = int(seq_s2[1] * 8) % 8 + 1
        operation_rule = 3  # FORCE XOR OPERATION
        decoding_rule = int(seq_s2[3] * 8) % 8 + 1
        
        # Generate key matrix
        segment = seq_y[:self.height * self.width]
        key_matrix = (segment.reshape((self.height, self.width)) * 255).astype(np.uint8)
        
        # Encode diffused image and key matrix
        encoded_image = self._dna_encode(diffused, encoding_rule)
        encoded_key = self._dna_encode(key_matrix, key_rule)
        
        # Perform DNA operation
        result_dna = self._dna_operation(encoded_image, encoded_key, operation_rule)
        
        # Decode the result
        encrypted = self._dna_decode(result_dna, decoding_rule)
        
        # Store sequences and DNA rules for decryption
        self.encryption_sequences[channel_idx] = {
            'seq_x': seq_x,
            'seq_y': seq_y,
            'seq_s1': seq_s1,
            'perm_indices': perm_indices
        }
        self.dna_operations[channel_idx] = {
            'encoding_rule': encoding_rule,
            'key_rule': key_rule,
            'operation_rule': operation_rule,
            'decoding_rule': decoding_rule
        }
        
        return encrypted
    
    def _process_channel_decrypt(self, channel_idx, encrypted_channel):
        """Process a single channel for decryption"""
        # Retrieve sequences and DNA rules used in encryption
        seqs = self.encryption_sequences[channel_idx]
        dna_ops = self.dna_operations[channel_idx]
        seq_x = seqs['seq_x']
        seq_y = seqs['seq_y']
        perm_indices = seqs['perm_indices']
        encoding_rule = dna_ops['encoding_rule']
        key_rule = dna_ops['key_rule']
        operation_rule = dna_ops['operation_rule']
        decoding_rule = dna_ops['decoding_rule']
        
        # Generate key matrix
        segment = seq_y[:self.height * self.width]
        key_matrix = (segment.reshape((self.height, self.width)) * 255).astype(np.uint8)
        
        # Reverse DNA operations (in reverse order)
        # 1. Encode encrypted image and key matrix
        encoded_encrypted = self._dna_encode(encrypted_channel, decoding_rule)
        encoded_key = self._dna_encode(key_matrix, key_rule)
        
        # 2. Reverse DNA operation - ALWAYS USE XOR (self-inverse)
        inverse_operation = 3  # Always use XOR
        result_dna = self._dna_operation(encoded_encrypted, encoded_key, inverse_operation)
        
        # 3. Decode the result
        diffused = self._dna_decode(result_dna, encoding_rule)
        
        # 4. Reverse 1D Diffusion
        scrambled = self._reverse_1d_diffusion(diffused, seq_x)
        
        # 5. Reverse global scrambling
        permuted = self._reverse_global_scrambling(scrambled, perm_indices)
        
        # 6. Reverse 3D Permutation
        decrypted = self._inverse_3d_permutation_vectorized(permuted)
        
        return decrypted
    
    def encrypt(self):
        """Encrypt the image with parallel channel processing"""
        start_time = time.time()
        
        # Generate keys only if not already generated
        if self.A is None or self.B is None:
            _, self.A, self.B = self._generate_keys(self.original_image.tobytes())
        
        results = []
        for ch in [0, 1, 2]:
            results.append(self._process_channel_encrypt(ch))
        
        self.encrypted_image = np.stack(results, axis=-1)
        self.encryption_time = time.time() - start_time
        return self.encrypted_image
    
    def decrypt(self, encrypted_image=None):
        """Decrypt the image"""
        start_time = time.time()
        if encrypted_image is None:
            encrypted_image = self.encrypted_image
        
        # Verify we have encryption sequences
        if not self.encryption_sequences:
            raise RuntimeError("Must encrypt before decrypting")
            
        channels = []
        for ch in range(3):
            decrypted_ch = self._process_channel_decrypt(ch, encrypted_image[:, :, ch])
            channels.append(decrypted_ch)
        
        self.decrypted_image = np.stack(channels, axis=-1)
        self.decryption_time = time.time() - start_time
        
        return self.decrypted_image
    
    def add_noise(self, image, noise_type, level):
        """Add noise to image with correct parameters"""
        # Convert to float in [0,1] range
        image_float = image.astype(float) / 255.0
        
        if noise_type == 's&p':
            noisy = random_noise(image_float, mode='s&p', amount=level)
        elif noise_type == 'gaussian':
            noisy = random_noise(image_float, mode='gaussian', var=level)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        return (noisy * 255).astype(np.uint8)
    
    def plot_noise_attacks(self, noise_type, levels):
        """Create separate plot for noise attacks"""
        n = len(levels)
        fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
        fig.suptitle(f'{noise_type.capitalize()} Noise Attack Analysis', fontsize=16)
        
        # Process noise attacks
        for i, level in enumerate(levels):
            # Add noise to encrypted image
            noisy_enc = self.add_noise(self.encrypted_image, noise_type, level)
            
            # Decrypt noisy image
            decrypted_noisy = self.decrypt(noisy_enc)
            
            # Calculate PSNR
            psnr_val = psnr(self.original_image_backup, decrypted_noisy, data_range=255)
            
            # Save individual image
            cv2.imwrite(f'results/{noise_type}_noise_{level}_decrypted.png', decrypted_noisy)
            
            # Display results
            if n > 1:
                ax0 = axes[0, i]
                ax1 = axes[1, i]
            else:
                ax0 = axes[0]
                ax1 = axes[1]
                
            ax0.imshow(cv2.cvtColor(noisy_enc, cv2.COLOR_BGR2RGB))
            ax0.set_title(f'Noisy Cipher\n({noise_type}: {level})')
            ax0.axis('off')
            
            ax1.imshow(cv2.cvtColor(decrypted_noisy, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'Decrypted Image\nPSNR: {psnr_val:.2f} dB')
            ax1.axis('off')
            
            # Add PSNR to results
            self.results.setdefault('noise_attack', {}).setdefault(noise_type, []).append(psnr_val)
        
        plt.tight_layout()
        plt.savefig(f'results/noise_attack_{noise_type}.png')
        plt.close()
    
    def verify_decryption(self):
        """Verify that decryption works correctly without noise"""
        # Decrypt the clean encrypted image
        clean_decrypted = self.decrypt()
        
        # Calculate PSNR and SSIM
        psnr_val = psnr(self.original_image_backup, clean_decrypted, data_range=255)
        ssim_val = ssim(
            self.original_image_backup, 
            clean_decrypted, 
            multichannel=True, 
            channel_axis=2,
            data_range=255
        )
        
        # Save images
        cv2.imwrite('results/clean_decrypted.png', clean_decrypted)
        
        # Visual comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(self.original_image_backup, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(self.encrypted_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Encrypted Image')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(clean_decrypted, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Decrypted Image\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/decryption_verification.png')
        plt.close()
        
        # Add to results
        self.results['clean_decryption'] = {
            'psnr': psnr_val,
            'ssim': ssim_val
        }
    
    def correlation_analysis(self):
        """Calculate and plot pixel correlations for encrypted and decrypted images"""
        # Encrypted image correlations
        fig_enc, axes_enc = plt.subplots(3, 3, figsize=(18, 18))
        fig_enc.suptitle('Encrypted Image Correlation Analysis', fontsize=20)
        
        # Decrypted image correlations
        clean_decrypted = self.decrypt()
        fig_dec, axes_dec = plt.subplots(3, 3, figsize=(18, 18))
        fig_dec.suptitle('Decrypted Image Correlation Analysis', fontsize=20)
        
        directions = ['Horizontal', 'Vertical', 'Diagonal']
        colors = ['Red', 'Green', 'Blue']
        plot_colors = ['red', 'green', 'blue']
        
        for ch, (color, pcolor) in enumerate(zip(colors, plot_colors)):
            # Encrypted image
            enc_ch = self.encrypted_image[:, :, ch].flatten()
            
            # Decrypted image
            dec_ch = clean_decrypted[:, :, ch].flatten()
            
            for i, direction in enumerate(directions):
                # Encrypted image correlations
                enc_pairs = self._get_pixel_pairs(enc_ch, direction)
                enc_coeff = self._calculate_correlation(enc_pairs)
                
                # Plot encrypted
                ax_enc = axes_enc[ch, i]
                ax_enc.scatter(enc_pairs[:, 0], enc_pairs[:, 1], s=1, alpha=0.3, c=pcolor)
                ax_enc.set_title(f'Encrypted {color} Channel\n{direction} Correlation (r={enc_coeff:.4f})')
                ax_enc.set_xlabel('Pixel Intensity')
                ax_enc.set_ylabel('Adjacent Pixel Intensity')
                ax_enc.set_xlim(0, 255)
                ax_enc.set_ylim(0, 255)
                
                # Decrypted image correlations
                dec_pairs = self._get_pixel_pairs(dec_ch, direction)
                dec_coeff = self._calculate_correlation(dec_pairs)
                
                # Plot decrypted
                ax_dec = axes_dec[ch, i]
                ax_dec.scatter(dec_pairs[:, 0], dec_pairs[:, 1], s=1, alpha=0.3, c=pcolor)
                ax_dec.set_title(f'Decrypted {color} Channel\n{direction} Correlation (r={dec_coeff:.4f})')
                ax_dec.set_xlabel('Pixel Intensity')
                ax_dec.set_ylabel('Adjacent Pixel Intensity')
                ax_dec.set_xlim(0, 255)
                ax_dec.set_ylim(0, 255)
        
        plt.figure(fig_enc.number)
        plt.tight_layout()
        plt.savefig('results/correlation_analysis_encrypted.png')
        plt.close(fig_enc)
        
        plt.figure(fig_dec.number)
        plt.tight_layout()
        plt.savefig('results/correlation_analysis_decrypted.png')
        plt.close(fig_dec)
        
        # Save correlation coefficients
        self.results['correlation_coefficients'] = {
            'encrypted': enc_coeff,
            'decrypted': dec_coeff
        }
    
    def _get_pixel_pairs(self, channel, direction):
        """Get pixel pairs for correlation analysis"""
        h, w = self.height, self.width
        flat = channel
        idx = np.random.choice(len(flat) - w - 1, 5000, replace=False)
        
        if direction == 'Horizontal':
            pairs = np.vstack((flat[idx], flat[idx + 1])).T
        elif direction == 'Vertical':
            pairs = np.vstack((flat[idx], flat[idx + w])).T
        else:  # Diagonal
            pairs = np.vstack((flat[idx], flat[idx + w + 1])).T
        
        return pairs
    
    def _calculate_correlation(self, pairs):
        """Calculate correlation coefficient"""
        x, y = pairs[:, 0], pairs[:, 1]
        cov = np.cov(x, y)[0, 1]
        std_x, std_y = np.std(x), np.std(y)
        return cov / (std_x * std_y) if std_x * std_y != 0 else 0
    
    def histogram_analysis(self):
        """Plot histograms of original and encrypted images"""
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        colors = ['Red', 'Green', 'Blue']
        
        for i, color in enumerate(colors):
            # Original image histogram
            axes[i, 0].hist(
                self.original_image_backup[:, :, i].ravel(), 
                bins=256, color=color.lower(), alpha=0.7
            )
            axes[i, 0].set_title(f'Original Image - {color} Channel')
            axes[i, 0].set_xlim([0, 256])
            
            # Encrypted image histogram
            axes[i, 1].hist(
                self.encrypted_image[:, :, i].ravel(), 
                bins=256, color='gray', alpha=0.7
            )
            axes[i, 1].set_title(f'Encrypted Image - {color} Channel')
            axes[i, 1].set_xlim([0, 256])
        
        plt.tight_layout()
        plt.savefig('results/histogram_analysis.png')
        plt.close()
    
    def calculate_entropy(self):
        """Calculate and compare image entropy"""
        def channel_entropy(channel):
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            hist = hist[hist > 0] / channel.size
            return -np.sum(hist * np.log2(hist))
        
        orig_entropy = [channel_entropy(self.original_image_backup[:, :, i]) for i in range(3)]
        enc_entropy = [channel_entropy(self.encrypted_image[:, :, i]) for i in range(3)]
        
        self.results['entropy'] = {
            'original': orig_entropy,
            'encrypted': enc_entropy
        }
        
        print("Entropy Results:")
        print(f"Original Image: R={orig_entropy[0]:.4f}, G={orig_entropy[1]:.4f}, B={orig_entropy[2]:.4f}")
        print(f"Encrypted Image: R={enc_entropy[0]:.4f}, G={enc_entropy[1]:.4f}, B={enc_entropy[2]:.4f}")
    
    def calculate_npcr_uaci(self):
        """Calculate NPCR and UACI metrics"""
        # Create modified original image (change one pixel)
        modified_orig = self.original_image_backup.copy()
        modified_orig[0, 0, 0] = (modified_orig[0, 0, 0] + 1) % 256
        
        # Create temporary encryptor for modified image
        temp_encryptor = ImageEncryptor(self.image_path)
        temp_encryptor.original_image = modified_orig
        temp_encryptor.original_image_backup = modified_orig
        mod_enc = temp_encryptor.encrypt()
        
        # Calculate NPCR and UACI
        total_pixels = self.encrypted_image.size
        diff = (self.encrypted_image != mod_enc).astype(int)
        npcr = np.sum(diff) / total_pixels * 100
        
        abs_diff = np.abs(self.encrypted_image.astype(int) - mod_enc.astype(int))
        uaci = np.sum(abs_diff) / (255 * total_pixels) * 100
        
        self.results['npcr'] = npcr
        self.results['uaci'] = uaci
        
        print(f"NPCR: {npcr:.4f}% (Ideal: 99.6094%)")
        print(f"UACI: {uaci:.4f}% (Ideal: 33.4635%)")
        
        # Save images for reference
        cv2.imwrite('results/original_cipher.png', self.encrypted_image)
        cv2.imwrite('results/modified_cipher.png', mod_enc)
    
    def shear_attack_analysis(self):
        """Perform shear attack analysis with random pixel corruption"""
        attacks = [
            ('top_left_16', lambda img: self._shear_top_left(img, fraction=1/16)),
            ('top_left_25', lambda img: self._shear_top_left(img, fraction=0.25)),
            ('top_half', lambda img: self._shear_top_half(img)),
            ('middle', lambda img: self._shear_middle(img))
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Shear Attack Analysis', fontsize=16)
        
        for i, (attack_name, attack_func) in enumerate(attacks):
            # Apply shear to encrypted image
            sheared_enc = attack_func(self.encrypted_image.copy())
            
            # Decrypt sheared image
            decrypted_sheared = self.decrypt(sheared_enc)
            
            # Save images for reference
            cv2.imwrite(f'results/shear_{attack_name}_encrypted.png', sheared_enc)
            cv2.imwrite(f'results/shear_{attack_name}_decrypted.png', decrypted_sheared)
            
            # Plot encrypted row (row 0)
            axes[0, i].imshow(cv2.cvtColor(sheared_enc, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Sheared Cipher\n({attack_name.replace("_", " ")})')
            axes[0, i].axis('off')
            
            # Plot decrypted row (row 1)
            axes[1, i].imshow(cv2.cvtColor(decrypted_sheared, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f'Decrypted Image')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/shear_attack_analysis.png')
        plt.close()
    
    def _shear_top_left(self, img, fraction=0.25):
        """Shear top left corner with random noise"""
        h, w, c = img.shape
        h_shear = int(h * fraction)
        w_shear = int(w * fraction)
        img[:h_shear, :w_shear] = np.random.randint(0, 256, (h_shear, w_shear, c), dtype=np.uint8)
        return img
    
    def _shear_top_half(self, img):
        """Shear top half with random noise"""
        h, w, c = img.shape
        img[:h//2, :] = np.random.randint(0, 256, (h//2, w, c), dtype=np.uint8)
        return img
    
    def _shear_middle(self, img):
        """Shear middle part with random noise"""
        h, w, c = img.shape
        h_start, h_end = h//4, 3*h//4
        w_start, w_end = w//4, 3*w//4
        img[h_start:h_end, w_start:w_end] = np.random.randint(
            0, 256, (h_end - h_start, w_end - w_start, c), dtype=np.uint8
        )
        return img
    
    def key_sensitivity_analysis(self):
        """Test key sensitivity with significant key modification"""
        print("\n===== Key Sensitivity Analysis =====")
        print("Original Keys:")
        print(f"A: {self.A}")
        print(f"B: {self.B}")
        
        # Create a sensitive encryptor
        sensitive_encryptor = ImageEncryptor(self.image_path)
        sensitive_encryptor.original_image = self.original_image_backup.copy()
        sensitive_encryptor.row_index = self.row_index
        sensitive_encryptor.col_index = self.col_index
        
        # Generate modified keys (significant modification)
        modified_A = self.A.copy()
        modified_A[0] += 0.1  # Significant modification
        
        print("\nModified Keys:")
        print(f"A: {modified_A}")
        print(f"B: {self.B}")
        
        # Set modified keys
        sensitive_encryptor.A = modified_A
        sensitive_encryptor.B = self.B
        
        # Encrypt with modified keys
        modified_cipher = sensitive_encryptor.encrypt()
        
        # Attempt decryption with modified keys
        sensitive_encryptor.A = modified_A  # Keep modified key for decryption
        wrong_decrypted = sensitive_encryptor.decrypt(modified_cipher)
        
        # Correct decryption with original keys
        self.decrypt(self.encrypted_image)
        correct_decrypted = self.decrypted_image.copy()
        
        # Calculate PSNR
        psnr_val = psnr(correct_decrypted, wrong_decrypted, data_range=255)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(correct_decrypted, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Correct Key Decryption')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(wrong_decrypted, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Modified Key Decryption\nPSNR: {psnr_val:.2f} dB')
        axes[1].axis('off')
        
        # Add key info to plot
        plt.figtext(0.5, 0.01, 
                   f"Original A: {self.A[:2]}...\nModified A: {modified_A[:2]}...", 
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('results/key_sensitivity_analysis.png')
        plt.close()
        
        self.results['key_sensitivity_psnr'] = psnr_val
        print(f"Key Sensitivity PSNR: {psnr_val:.2f} dB")
    
    def run_full_analysis(self):
        """Run all analysis steps with progress tracking"""
        steps = [
            ("Encrypting image", self.encrypt),
            ("Saving encrypted image", lambda: cv2.imwrite('results/encrypted_image.png', self.encrypted_image)),
            ("Verifying clean decryption", self.verify_decryption),
            ("Performing salt & pepper noise attack analysis", lambda: self.plot_noise_attacks('s&p', [0.01, 0.02, 0.1, 0.2])),
            ("Performing Gaussian noise attack analysis", lambda: self.plot_noise_attacks('gaussian', [0.01, 0.02, 0.1, 0.2])),
            ("Performing shear attack analysis", self.shear_attack_analysis),
            ("Performing key sensitivity analysis", self.key_sensitivity_analysis),
            ("Calculating NPCR and UACI", self.calculate_npcr_uaci),
            ("Performing correlation analysis", self.correlation_analysis),
            ("Performing histogram analysis", self.histogram_analysis),
            ("Calculating entropy", self.calculate_entropy),
            ("Saving results", lambda: json.dump(self.results, open('results/full_analysis.json', 'w'), indent=4))
        ]
        
        # Print timing information
        print("\n=== Timing Summary ===")
        print(f"Key generation time: {self.key_gen_time:.4f} seconds")
        print(f"Encryption time: {self.encryption_time:.4f} seconds")
        print(f"Decryption time: {self.decryption_time:.4f} seconds")
        print(f"DNA operations time: {self.dna_time:.4f} seconds")
        
        # Add to results
        self.results['timing'] = {
            'key_generation': self.key_gen_time,
            'encryption': self.encryption_time,
            'decryption': self.decryption_time,
            'dna_operations': self.dna_time
        }
        
        for desc, task in tqdm(steps, desc="Processing"):
            print(f"\n{desc}...")
            task()
        
        print("Analysis complete! Results saved in 'results' directory.")

# Main execution
if __name__ == "__main__":
    encryptor = ImageEncryptor('/kaggle/input/encryption-images/mandrill.png')
    encryptor.run_full_analysis()