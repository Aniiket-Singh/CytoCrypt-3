# dna_image_cipher.py
import numpy as np
import cv2
import json
import math
import os
import sys
import time
import argparse
from typing import Tuple, Dict, List, Union
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ===================================================================
# 1. Enhanced Chaotic Map System
# ===================================================================
class EnhancedChaoticSystem:
    """Manages all chaotic maps with support for multiple instances"""
    def __init__(self, config: Dict):
        self.config = config
        self.transient_iter = config.get("transient_iterations", 1000)
        
    def logistic_map(self, length: int, x0: float, r: float) -> np.ndarray:
        seq = np.zeros(length)
        x = x0
        for _ in range(self.transient_iter):
            x = r * x * (1 - x)
        for i in range(length):
            x = r * x * (1 - x)
            seq[i] = x
        return seq
    
    def sine_map(self, length: int, x0: float, a: float) -> np.ndarray:
        seq = np.zeros(length)
        x = x0
        for _ in range(self.transient_iter):
            x = a * math.sin(math.pi * x)
        for i in range(length):
            x = a * math.sin(math.pi * x)
            seq[i] = x
        return seq
    
    def singer_map(self, length: int, x0: float, mu: float) -> np.ndarray:
        seq = np.zeros(length)
        x = x0
        for _ in range(self.transient_iter):
            x = mu * (7.86*x - 23.31*(x**2) + 28.75*(x**3) - 13.3*(x**4))
        for i in range(length):
            x = mu * (7.86*x - 23.31*(x**2) + 28.75*(x**3) - 13.3*(x**4))
            seq[i] = x
        return seq
    
    def quadratic_map(self, length: int, x0: float, a: float) -> np.ndarray:
        seq = np.zeros(length)
        x = x0
        for _ in range(self.transient_iter):
            x = a - x**2
        for i in range(length):
            x = a - x**2
            seq[i] = x
        return seq
    
    def pwlcm_map(self, length: int, x0: float, p: float) -> np.ndarray:
        """Piecewise Linear Chaotic Map"""
        seq = np.zeros(length)
        x = x0
        for _ in range(self.transient_iter):
            if x < p:
                x = x / p
            elif x < 0.5:
                x = (x - p) / (0.5 - p)
            else:
                x = 1.0 - x
        for i in range(length):
            if x < p:
                x = x / p
            elif x < 0.5:
                x = (x - p) / (0.5 - p)
            else:
                x = 1.0 - x
            seq[i] = x
        return seq
    
    def lorenz_map(self, length: int, x0: float, y0: float, z0: float, 
                  sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3) -> np.ndarray:
        """Lorenz system (returns x-component)"""
        seq = np.zeros(length)
        x, y, z = x0, y0, z0
        dt = 0.01
        for _ in range(self.transient_iter):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz
        for i in range(length):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz
            seq[i] = x
        return seq
    
    def ikeda_map(self, length: int, x0: float, y0: float, u: float = 0.6) -> np.ndarray:
        """Ikeda map (returns x-component)"""
        seq = np.zeros(length)
        x, y = x0, y0
        for _ in range(self.transient_iter):
            t = 0.4 - 6.0 / (1 + x*x + y*y)
            x1 = 1 + u * (x * math.cos(t) - y * math.sin(t))
            y = u * (x * math.sin(t) + y * math.cos(t))
            x = x1
        for i in range(length):
            t = 0.4 - 6.0 / (1 + x*x + y*y)
            x1 = 1 + u * (x * math.cos(t) - y * math.sin(t))
            y = u * (x * math.sin(t) + y * math.cos(t))
            x = x1
            seq[i] = x
        return seq
    
    def henon_map(self, length: int, x0: float, y0: float, a: float = 1.4, b: float = 0.3) -> np.ndarray:
        """Henon map (returns x-component)"""
        seq = np.zeros(length)
        x, y = x0, y0
        for _ in range(self.transient_iter):
            x1 = 1 - a * x*x + y
            y = b * x
            x = x1
        for i in range(length):
            x1 = 1 - a * x*x + y
            y = b * x
            x = x1
            seq[i] = x
        return seq
    
    def tent_map(self, length: int, x0: float, b: float = 2.0) -> np.ndarray:
        """Tent map"""
        seq = np.zeros(length)
        x = x0
        for _ in range(self.transient_iter):
            if x < 0.5:
                x = b * x
            else:
                x = b * (1 - x)
        for i in range(length):
            if x < 0.5:
                x = b * x
            else:
                x = b * (1 - x)
            seq[i] = x
        return seq
    
    def generate_sequence(self, map_type: str, length: int, params: Dict) -> np.ndarray:
        """Unified chaotic sequence generator with multi-instance support"""
        # Handle multiple logistic/sine instances
        if map_type.startswith("logistic"):
            return self.logistic_map(length, params["x0"], params["r"])
        elif map_type.startswith("sine"):
            return self.sine_map(length, params["x0"], params["a"])
        
        # Other map types
        if map_type == "singer":
            return self.singer_map(length, params["x0"], params["mu"])
        elif map_type == "quadratic":
            return self.quadratic_map(length, params["x0"], params["a"])
        elif map_type == "pwlcm":
            return self.pwlcm_map(length, params["x0"], params["p"])
        elif map_type == "lorenz":
            return self.lorenz_map(length, params["x0"], params["y0"], params["z0"])
        elif map_type == "ikeda":
            return self.ikeda_map(length, params["x0"], params["y0"], params["u"])
        elif map_type == "henon":
            return self.henon_map(length, params["x0"], params["y0"], params["a"], params["b"])
        elif map_type == "tent":
            return self.tent_map(length, params["x0"], params["b"])
        else:
            raise ValueError(f"Unknown map type: {map_type}")

# ===================================================================
# 2. DNA Encoding System
# ===================================================================
class DNAEncoder:
    """Handles DNA encoding/decoding with dynamic rules"""
    DNA_RULES = {
        1: {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
        2: {'00': 'A', '01': 'G', '10': 'C', '11': 'T'},
        3: {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},
        4: {'00': 'G', '01': 'A', '10': 'T', '11': 'C'},
        5: {'00': 'T', '01': 'C', '10': 'G', '11': 'A'},
        6: {'00': 'T', '01': 'G', '10': 'C', '11': 'A'},
        7: {'00': 'G', '01': 'C', '10': 'A', '11': 'T'},
        8: {'00': 'C', '01': 'G', '10': 'A', '11': 'T'}
    }
    
    DNA_REVERSE = {rule: {v: k for k, v in mapping.items()} for rule, mapping in DNA_RULES.items()}
    
    DNA_XOR = {
        'A': {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T'},
        'C': {'A': 'C', 'C': 'A', 'G': 'T', 'T': 'G'},
        'G': {'A': 'G', 'C': 'T', 'G': 'A', 'T': 'C'},
        'T': {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    }
    
    @staticmethod
    def encode_image(image: np.ndarray, rule_seq: np.ndarray) -> np.ndarray:
        bin_img = np.vectorize(lambda x: np.binary_repr(x, width=8))(image)
        dna_matrix = np.empty((*image.shape, 4), dtype='U1')
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # FIXED: Properly scale to 8 rules
                rule_val = rule_seq[i, j]
                rule = int(rule_val * 8) % 8 + 1
                pixel_bin = bin_img[i, j]
                bases = [
                    DNAEncoder.DNA_RULES[rule][pixel_bin[k:k+2]] 
                    for k in range(0, 8, 2)
                ]
                dna_matrix[i, j] = bases
        return dna_matrix
    
    @staticmethod
    def decode_image(dna_matrix: np.ndarray, rule_seq: np.ndarray) -> np.ndarray:
        decoded = np.zeros(dna_matrix.shape[:2], dtype=np.uint8)
        
        for i in range(dna_matrix.shape[0]):
            for j in range(dna_matrix.shape[1]):
                # FIXED: Properly scale to 8 rules
                rule_val = rule_seq[i, j]
                rule = int(rule_val * 8) % 8 + 1
                bases = dna_matrix[i, j]
                bin_str = ""
                for base in bases:
                    bin_str += DNAEncoder.DNA_REVERSE[rule][base]
                decoded[i, j] = int(bin_str, 2)
        return decoded
    
    @staticmethod
    def dna_xor(dna_matrix1: np.ndarray, dna_matrix2: np.ndarray) -> np.ndarray:
        result = np.empty_like(dna_matrix1)
        for i in range(dna_matrix1.shape[0]):
            for j in range(dna_matrix1.shape[1]):
                for k in range(4):
                    base1 = dna_matrix1[i, j, k]
                    base2 = dna_matrix2[i, j, k]
                    result[i, j, k] = DNAEncoder.DNA_XOR[base1][base2]
        return result

# ===================================================================
# 3. Enhanced Encryption/Decryption
# ===================================================================
class EnhancedDNAImageCipher:
    """Encryption/decryption with configurable rounds and avalanche test support"""
    def __init__(self, config_file: str = "config_enhanced.json"):
        self.config = self._load_config(config_file)
        self.chaos_sys = EnhancedChaoticSystem(self.config["chaos_params"])
        self.original_config = json.load(open(config_file))  # Save original config for avalanche
        
    @staticmethod
    def _load_config(file_path: str) -> Dict:
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {
                "chaos_params": {
                    "transient_iterations": 1000,
                    "logistic": {"x0": 0.345, "r": 3.89},
                    "sine": {"x0": 0.456, "a": 0.99},
                    "singer": {"x0": 0.567, "mu": 1.07},
                    "quadratic": {"x0": 0.123, "a": 1.75},
                    "pwlcm": {"x0": 0.234, "p": 0.25},
                    "lorenz": {"x0": 0.1, "y0": 0.2, "z0": 0.3},
                    "ikeda": {"x0": 0.345, "y0": 0.456, "u": 0.6},
                    "henon": {"x0": 0.1, "y0": 0.2, "a": 1.4, "b": 0.3},
                    "tent": {"x0": 0.123, "b": 2.0},
                    "logistic2": {"x0": 0.789, "r": 3.999},
                    "sine2": {"x0": 0.891, "a": 0.95},
                },
                "scrambling_maps": [
                    "quadratic", "pwlcm", "lorenz", "ikeda",
                    "henon", "tent", "logistic2", "sine2"
                ]
            }
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4)
        return config
    
    def _xor_operation(self, image: np.ndarray, k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
        # Convert to uint16 to prevent overflow during addition
        k1_16 = k1.astype(np.uint16)
        k2_16 = k2.astype(np.uint16)
        
        # Perform addition and modulo operation in uint16 space
        combined = (k1_16 + k2_16) % 256
        
        # Convert back to uint8 before XOR operation
        combined_uint8 = combined.astype(np.uint8)
        return np.bitwise_xor(image, combined_uint8)
    
    def _generate_key(self, shape: Tuple[int, int], map_type: str) -> np.ndarray:
        params = self.config["chaos_params"][map_type]
        seq = self.chaos_sys.generate_sequence(map_type, shape[0]*shape[1], params)
        
        # Convert to binary key with better sensitivity
        binary_key = np.zeros_like(seq, dtype=np.uint8)
        for i in range(len(seq)):
            # Use multiple thresholds for better diffusion
            if seq[i] < 0.25:
                binary_key[i] = 0
            elif seq[i] < 0.5:
                binary_key[i] = 1
            elif seq[i] < 0.75:
                binary_key[i] = 2
            else:
                binary_key[i] = 3
        
        return binary_key.reshape(shape)

    
    def _generate_rule_seq(self, shape: Tuple[int, int]) -> np.ndarray:
        params = self.config["chaos_params"]["logistic"]
        seq = self.chaos_sys.generate_sequence("logistic", shape[0]*shape[1], params)
        return seq.reshape(shape)
    
    def _generate_dna_key(self, shape: Tuple[int, int]) -> np.ndarray:
        # Generate key image values
        params = self.config["chaos_params"]["sine"]
        seq = self.chaos_sys.generate_sequence("sine", shape[0]*shape[1], params)
        img_seq = (seq * 255).astype(np.uint8).reshape(shape)
        
        # Generate rule sequence as FLOAT array (0-1)
        rule_params = self.config["chaos_params"]["logistic"]
        rule_seq = self.chaos_sys.generate_sequence("logistic", shape[0]*shape[1], rule_params)
        rule_seq = rule_seq.reshape(shape)
        
        return DNAEncoder.encode_image(img_seq, rule_seq)
    
    def _split_blocks(self, image: np.ndarray) -> List[np.ndarray]:
        h, w = image.shape
        block_h, block_w = h // 4, w // 4
        blocks = []
        
        for i in range(4):
            for j in range(4):
                y_start = i * block_h
                y_end = (i+1) * block_h if i < 3 else h
                x_start = j * block_w
                x_end = (j+1) * block_w if j < 3 else w
                blocks.append(image[y_start:y_end, x_start:x_end])
        return blocks
    
    def _merge_blocks(self, blocks: List[np.ndarray], orig_shape: Tuple[int, int]) -> np.ndarray:
        h, w = orig_shape
        block_h, block_w = h // 4, w // 4
        rows = []
        
        for i in range(4):
            row_blocks = []
            for j in range(4):
                idx = i*4 + j
                if j == 3:
                    block_w_adjusted = w - 3*block_w
                else:
                    block_w_adjusted = block_w
                    
                if i == 3:
                    block_h_adjusted = h - 3*block_h
                else:
                    block_h_adjusted = block_h
                
                if blocks[idx].shape != (block_h_adjusted, block_w_adjusted):
                    blocks[idx] = blocks[idx][:block_h_adjusted, :block_w_adjusted]
                
                row_blocks.append(blocks[idx])
            rows.append(np.concatenate(row_blocks, axis=1))
        return np.concatenate(rows, axis=0)
    
    def _scramble_block(self, block: np.ndarray, map_type: str) -> np.ndarray:
        params = self.config["chaos_params"][map_type]
        seq = self.chaos_sys.generate_sequence(map_type, block.size, params)
        perm = np.argsort(seq)
        flat_block = block.flatten()
        scrambled = flat_block[perm].reshape(block.shape)
        return scrambled
    
    def _unscramble_block(self, block: np.ndarray, map_type: str) -> np.ndarray:
        params = self.config["chaos_params"][map_type]
        seq = self.chaos_sys.generate_sequence(map_type, block.size, params)
        perm = np.argsort(seq)
        rev_perm = np.argsort(perm)
        flat_block = block.flatten()
        unscrambled = flat_block[rev_perm].reshape(block.shape)
        return unscrambled
    
    def _scramble_blocks(self, image: np.ndarray) -> np.ndarray:
        singer_params = self.config["chaos_params"]["singer"]
        singer_seq = self.chaos_sys.generate_sequence("singer", 16, singer_params)
        
        min_val, max_val = np.min(singer_seq), np.max(singer_seq)
        singer_seq = (singer_seq - min_val) / (max_val - min_val)
        map_indices = (singer_seq * 8).astype(int) % 8
        scrambling_maps = self.config["scrambling_maps"]
        
        blocks = self._split_blocks(image)
        orig_shape = image.shape
        scrambled_blocks = []
        
        total_blocks = 16
        for i in range(total_blocks):
            map_idx = map_indices[i]
            map_name = scrambling_maps[map_idx]
            scrambled = self._scramble_block(blocks[i], map_name)
            scrambled_blocks.append(scrambled)
        return self._merge_blocks(scrambled_blocks, orig_shape)
    
    def _unscramble_blocks(self, image: np.ndarray) -> np.ndarray:
        singer_params = self.config["chaos_params"]["singer"]
        singer_seq = self.chaos_sys.generate_sequence("singer", 16, singer_params)
        
        min_val, max_val = np.min(singer_seq), np.max(singer_seq)
        singer_seq = (singer_seq - min_val) / (max_val - min_val)
        map_indices = (singer_seq * 8).astype(int) % 8
        scrambling_maps = self.config["scrambling_maps"]
        
        blocks = self._split_blocks(image)
        orig_shape = image.shape
        unscrambled_blocks = []
        
        total_blocks = 16
        for i in range(total_blocks):
            map_idx = map_indices[i]
            map_name = scrambling_maps[map_idx]
            unscrambled = self._unscramble_block(blocks[i], map_name)
            unscrambled_blocks.append(unscrambled)
        return self._merge_blocks(unscrambled_blocks, orig_shape)
    
    def _encrypt_round(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        
        print("Generating logistic key...")
        k1 = self._generate_key(shape, "logistic")
        print("Generating sine key...")
        k2 = self._generate_key(shape, "sine")
        
        print("Performing XOR operation...")
        I2 = self._xor_operation(image, k1, k2)
        print("Generating rule sequence...")
        rule_seq = self._generate_rule_seq(shape)
        print("DNA encoding...")
        dna_img = DNAEncoder.encode_image(I2, rule_seq)
        print("Generating DNA key...")
        dna_key = self._generate_dna_key(shape)
        print("DNA XOR operation...")
        dna_xored = DNAEncoder.dna_xor(dna_img, dna_key)
        print("DNA decoding...")
        I3 = DNAEncoder.decode_image(dna_xored, rule_seq)
        print("Performing XOR operation...")
        I4 = self._xor_operation(I3, k1, k2)
        print("Scrambling blocks...")
        I5 = self._scramble_blocks(I4)
        print("Performing final XOR operation...")
        I6 = self._xor_operation(I5, k1, k2)
        
        return I6
    
    def _decrypt_round(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        
        print("Generating logistic key...")
        k1 = self._generate_key(shape, "logistic")
        print("Generating sine key...")
        k2 = self._generate_key(shape, "sine")
        
        print("Performing XOR operation...")
        I5 = self._xor_operation(image, k1, k2)
        print("Unscrambling blocks...")
        I4 = self._unscramble_blocks(I5)
        print("Performing XOR operation...")
        I3 = self._xor_operation(I4, k1, k2)
        print("Generating rule sequence...")
        rule_seq = self._generate_rule_seq(shape)
        print("DNA encoding...")
        dna_img = DNAEncoder.encode_image(I3, rule_seq)
        print("Generating DNA key...")
        dna_key = self._generate_dna_key(shape)
        print("DNA XOR operation...")
        dna_xored = DNAEncoder.dna_xor(dna_img, dna_key)
        print("DNA decoding...")
        I2 = DNAEncoder.decode_image(dna_xored, rule_seq)
        print("Performing final XOR operation...")
        I1 = self._xor_operation(I2, k1, k2)
        
        return I1
    
    def encrypt(self, image_path: str, output_path: str = "encrypted.png", rounds: int = 1):
        start_time = time.time()
        print(f"Loading image: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not found or invalid format")
        
        encrypted_img = img.copy()
        for i in range(rounds):
            round_start = time.time()
            print(f"\n=== ENCRYPTION ROUND {i+1}/{rounds} ===")
            encrypted_img = self._encrypt_round(encrypted_img)
            
            round_time = time.time() - round_start
            print(f"Round {i+1} completed in {round_time:.2f} seconds")
        
        print(f"Saving encrypted image: {output_path}")
        cv2.imwrite(output_path, encrypted_img)
        
        total_time = time.time() - start_time
        print(f"\nTotal encryption time: {total_time:.2f} seconds")
        return encrypted_img
    
    def decrypt(self, image_path: str, output_path: str = "decrypted.png", 
                rounds: int = 1, save: bool = True):
        start_time = time.time()
        print(f"Loading encrypted image: {image_path}")
        enc_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if enc_img is None:
            raise ValueError("Encrypted image not found")
        
        decrypted_img = enc_img.copy()
        for i in range(rounds):
            round_start = time.time()
            print(f"\n=== DECRYPTION ROUND {i+1}/{rounds} ===")
            decrypted_img = self._decrypt_round(decrypted_img)
            
            round_time = time.time() - round_start
            print(f"Round {i+1} completed in {round_time:.2f} seconds")
        
        if save:
            print(f"Saving decrypted image: {output_path}")
            cv2.imwrite(output_path, decrypted_img)
        
        total_time = time.time() - start_time
        print(f"\nTotal decryption time: {total_time:.2f} seconds")
        return decrypted_img

# ===================================================================
# 5. Command-Line Interface
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description='DNA Image Cipher Encryption/Decryption')
    parser.add_argument('mode', choices=['enc', 'dec'], help='Mode: enc for encryption, dec for decryption')
    parser.add_argument('rounds', type=int, help='Number of encryption/decryption rounds')
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('-o', '--output', help='Output image file path', default=None)
    
    args = parser.parse_args()
    
    print("Initializing cipher system...")
    cipher = EnhancedDNAImageCipher("config_enhanced.json")
    
    try:
        if args.mode == "enc":
            output = args.output or "encrypted.png"
            cipher.encrypt(args.input, output, args.rounds)
            print(f"\nEncryption complete! Output saved to {output}")
        
        elif args.mode == "dec":
            output = args.output or "decrypted.png"
            cipher.decrypt(args.input, output, args.rounds)
            print(f"\nDecryption complete! Output saved to {output}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()