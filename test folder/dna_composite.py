# dna_image_cipher.py
import numpy as np
import cv2
import json
import math
import os
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
# 3. Enhanced Encryption/Decryption with Rounds
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
    
# In EnhancedDNAImageCipher class:

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
        
        for i in range(16):
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
        
        for i in range(16):
            map_idx = map_indices[i]
            map_name = scrambling_maps[map_idx]
            unscrambled = self._unscramble_block(blocks[i], map_name)
            unscrambled_blocks.append(unscrambled)
        
        return self._merge_blocks(unscrambled_blocks, orig_shape)
    
    def _encrypt_round(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        k1 = self._generate_key(shape, "logistic")
        k2 = self._generate_key(shape, "sine")
        
        I2 = self._xor_operation(image, k1, k2)
        rule_seq = self._generate_rule_seq(shape)
        dna_img = DNAEncoder.encode_image(I2, rule_seq)
        dna_key = self._generate_dna_key(shape)
        dna_xored = DNAEncoder.dna_xor(dna_img, dna_key)
        I3 = DNAEncoder.decode_image(dna_xored, rule_seq)
        I4 = self._xor_operation(I3, k1, k2)
        I5 = self._scramble_blocks(I4)
        I6 = self._xor_operation(I5, k1, k2)
        
        return I6
    
    def _decrypt_round(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        k1 = self._generate_key(shape, "logistic")
        k2 = self._generate_key(shape, "sine")
        
        I5 = self._xor_operation(image, k1, k2)
        I4 = self._unscramble_blocks(I5)
        I3 = self._xor_operation(I4, k1, k2)
        rule_seq = self._generate_rule_seq(shape)
        dna_img = DNAEncoder.encode_image(I3, rule_seq)
        dna_key = self._generate_dna_key(shape)
        dna_xored = DNAEncoder.dna_xor(dna_img, dna_key)
        I2 = DNAEncoder.decode_image(dna_xored, rule_seq)
        I1 = self._xor_operation(I2, k1, k2)
        
        return I1
    
    def encrypt(self, image_path: str, output_path: str = "encrypted.png", rounds: int = 1):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not found or invalid format")
        
        encrypted_img = img.copy()
        for i in range(rounds):
            print(f"Encryption round {i+1}/{rounds}")
            encrypted_img = self._encrypt_round(encrypted_img)
        
        cv2.imwrite(output_path, encrypted_img)
        return encrypted_img
    
    # In dna_image_cipher.py, modify the decrypt method:
    def decrypt(self, image_path: str, output_path: str = "decrypted.png", 
                rounds: int = 1, save: bool = True):
        enc_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if enc_img is None:
            raise ValueError("Encrypted image not found")
        
        decrypted_img = enc_img.copy()
        for i in range(rounds):
            print(f"Decryption round {i+1}/{rounds}")
            decrypted_img = self._decrypt_round(decrypted_img)
        
        if save:
            cv2.imwrite(output_path, decrypted_img)
        return decrypted_img
    
    def generate_avalanche_images(self, image_path: str, output_base: str = "avalanche", rounds: int = 1):
        """Generate two encrypted images for avalanche test"""
        # 1. First encryption with original parameters
        orig_enc = self.encrypt(image_path, f"{output_base}_orig.png", rounds)
        
        # 2. Modify multiple parameters with more significant changes
        self.config["chaos_params"]["logistic"]["x0"] += 1e-5  # Increased from 1e-10
        self.config["chaos_params"]["sine"]["a"] += 1e-5       # Additional parameter change
        
        # 3. Second encryption with modified parameters
        mod_enc = self.encrypt(image_path, f"{output_base}_mod.png", rounds)
        
        # 4. Reset to original config
        self.config = json.loads(json.dumps(self.original_config))
        
        return orig_enc, mod_enc

# ===================================================================
# 4. Image Analysis Utilities
# ===================================================================
def load_image(path, mode='RGB'):
    img = Image.open(path).convert(mode)
    return np.array(img)

def is_grayscale(img_array):
    if len(img_array.shape) == 2:
        return True
    if len(img_array.shape) == 3:
        return (np.all(img_array[:, :, 0] == img_array[:, :, 1]) and 
                np.all(img_array[:, :, 0] == img_array[:, :, 2]))
    return False

def plot_histograms(orig, enc, save_path=None):
    if not is_grayscale(orig) or not is_grayscale(enc):
        print("Skipping histogram: One or both images are not grayscale")
        return
    
    if len(orig.shape) == 3:
        orig = orig[:, :, 0]
    if len(enc.shape) == 3:
        enc = enc[:, :, 0]
    
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image Histogram')
    plt.hist(orig.flatten(), bins=256, range=(0, 255), density=True)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.title('Encrypted Image Histogram')
    plt.hist(enc.flatten(), bins=256, range=(0, 255), density=True)
    plt.xlabel('Pixel value')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Histogram plot saved to {save_path}")
    else:
        plt.show()

def compute_npcr_uaci(enc1, enc2):
    assert enc1.shape == enc2.shape
    enc1_flat = enc1.astype(np.float32).flatten()
    enc2_flat = enc2.astype(np.float32).flatten()
    
    diff = enc1_flat != enc2_flat
    npcr = np.mean(diff) * 100
    
    abs_diff = np.abs(enc1_flat - enc2_flat)
    uaci = np.mean(abs_diff / 255) * 100
    
    return npcr, uaci

def correlation_coefficients(img):
    if len(img.shape) == 3:
        if is_grayscale(img):
            img = img[:, :, 0]
        else:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    h, w = img.shape
    horiz = img[:, :-1].flatten()
    horiz_neighbors = img[:, 1:].flatten()
    vert = img[:-1, :].flatten()
    vert_neighbors = img[1:, :].flatten()
    diag = img[:-1, :-1].flatten()
    diag_neighbors = img[1:, 1:].flatten()
    
    horiz_corr = pearsonr(horiz, horiz_neighbors)[0]
    vert_corr = pearsonr(vert, vert_neighbors)[0]
    diag_corr = pearsonr(diag, diag_neighbors)[0]
    
    return horiz_corr, vert_corr, diag_corr

def entropy(img):
    if len(img.shape) == 3:
        if is_grayscale(img):
            img = img[:, :, 0]
        else:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

# ===================================================================
# 5. Main Execution
# ===================================================================
def main():
    # =============================
    # USER CONFIGURATION
    # =============================
    IMAGE_PATH = "images/lena_grayscale.png"
    ROUNDS = 3  # Number of encryption/decryption rounds
    OUTPUT_DIR = "results"
    PERFORM_AVALANCHE_TEST = True  # Set to False to skip avalanche test
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize cipher
    cipher = EnhancedDNAImageCipher("config_enhanced.json")
    
    # =============================
    # ENCRYPTION
    # =============================
    print(f"\nEncrypting image with {ROUNDS} rounds...")
    encrypted = cipher.encrypt(
        IMAGE_PATH, 
        os.path.join(OUTPUT_DIR, "encrypted.png"), 
        rounds=ROUNDS
    )
    
    # =============================
    # DECRYPTION
    # =============================
    print(f"\nDecrypting image with {ROUNDS} rounds...")
    decrypted = cipher.decrypt(
        os.path.join(OUTPUT_DIR, "encrypted.png"), 
        os.path.join(OUTPUT_DIR, "decrypted.png"), 
        rounds=ROUNDS
    )
    
    # =============================
    # AVALANCHE TEST (Correct NPCR/UACI)
    # =============================
    if PERFORM_AVALANCHE_TEST:
        print("\nPerforming avalanche test...")
        enc_orig, enc_mod = cipher.generate_avalanche_images(
            IMAGE_PATH, 
            os.path.join(OUTPUT_DIR, "avalanche"), 
            rounds=ROUNDS
        )
        npcr, uaci = compute_npcr_uaci(enc_orig, enc_mod)
        print(f"NPCR (Avalanche): {npcr:.4f}%")
        print(f"UACI (Avalanche): {uaci:.4f}%")
        
        # Save avalanche images
        cv2.imwrite(os.path.join(OUTPUT_DIR, "avalanche_orig.png"), enc_orig)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "avalanche_mod.png"), enc_mod)
    
    # =============================
    # LOAD IMAGES FOR ANALYSIS
    # =============================
    orig_img = load_image(IMAGE_PATH, 'RGB')
    enc_img = load_image(os.path.join(OUTPUT_DIR, "encrypted.png"), 'RGB')
    dec_img = load_image(os.path.join(OUTPUT_DIR, "decrypted.png"), 'RGB')
    
    # =============================
    # ANALYSIS METRICS
    # =============================
    print("\nCalculating security metrics...")
    
    # 1. Histogram Analysis
    plot_histograms(
        orig_img, 
        enc_img, 
        save_path=os.path.join(OUTPUT_DIR, "histograms.png")
    )
    
    # 2. Correlation Coefficients
    cor_orig = correlation_coefficients(orig_img)
    cor_enc = correlation_coefficients(enc_img)
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    
    print("\nCorrelation Coefficients:")
    print("Original Image:")
    for d, c in zip(directions, cor_orig):
        print(f"  {d}: {c:.6f}")
    
    print("\nEncrypted Image:")
    for d, c in zip(directions, cor_enc):
        print(f"  {d}: {c:.6f}")
    
    # 3. Entropy
    ent_orig = entropy(orig_img)
    ent_enc = entropy(enc_img)
    print(f"\nEntropy:")
    print(f"  Original: {ent_orig:.6f} bits")
    print(f"  Encrypted: {ent_enc:.6f} bits")
    
    # 4. Decryption Accuracy
    orig_gray = np.array(Image.open(IMAGE_PATH).convert('L'))
    dec_gray = cv2.imread(os.path.join(OUTPUT_DIR, "decrypted.png"), cv2.IMREAD_GRAYSCALE)
    accuracy = np.mean(orig_gray == dec_gray) * 100
    print(f"\nDecryption Accuracy: {accuracy:.4f}%")
    
    
    print("\nAnalysis complete! Results saved to 'results' directory")

if __name__ == "__main__":
    main()