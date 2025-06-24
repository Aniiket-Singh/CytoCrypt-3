import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
import os
import json
from dna_image_cipher import EnhancedDNAImageCipher  # Import the cipher

# ================= CONFIGURATION =================
OUTPUT_DIR = "composite_rgb_v1"  # Change this to your desired output directory
PERFORM_AVALANCHE_TEST = True  # Set to False if you don't have avalanche images
ROUNDS = 1  # Must match the number of rounds used for encryption
# =================================================

def calculate_entropy(image):
    """Calculate Shannon entropy for both grayscale and RGB images"""
    if len(image.shape) == 3:  # RGB image
        entropies = []
        for c in range(3):
            hist = cv2.calcHist([image[:, :, c]], [0], None, [256], [0, 256])
            hist = hist.ravel() / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropies.append(entropy)
        return np.mean(entropies)
    else:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-10))

def plot_histograms(original, encrypted, save_path):
    """Plot histograms for original and encrypted images"""
    if len(original.shape) == 3 or len(encrypted.shape) == 3:
        # Handle color images
        fig = plt.figure(figsize=(15, 10))
        channels = ['Red', 'Green', 'Blue']
        
        # Original image histograms
        for i in range(3):
            plt.subplot(2, 3, i+1)
            plt.hist(original[:, :, i].ravel(), 256, [0, 256], color='red', alpha=0.7)
            plt.title(f'Original {channels[i]} Channel')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
        
        # Encrypted image histograms
        for i in range(3):
            plt.subplot(2, 3, i+4)
            plt.hist(encrypted[:, :, i].ravel(), 256, [0, 256], color='blue', alpha=0.7)
            plt.title(f'Encrypted {channels[i]} Channel')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
    else:
        # Handle grayscale images
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.hist(original.ravel(), 256, [0, 256], color='blue', alpha=0.7)
        plt.title('Original Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.subplot(122)
        plt.hist(encrypted.ravel(), 256, [0, 256], color='red', alpha=0.7)
        plt.title('Encrypted Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_correlation(image, direction='horizontal'):
    """Calculate correlation coefficients in specified direction"""
    if len(image.shape) == 3:  # RGB image
        corrs = []
        for c in range(3):
            corrs.append(_calculate_correlation_2d(image[:, :, c], direction))
        return np.mean(corrs)
    else:  # Grayscale
        return _calculate_correlation_2d(image, direction)

def _calculate_correlation_2d(image, direction='horizontal'):
    """Calculate correlation coefficients for a 2D image"""
    if direction == 'horizontal':
        pixels = image[:, :-1].flatten()
        adjacent = image[:, 1:].flatten()
    elif direction == 'vertical':
        pixels = image[:-1, :].flatten()
        adjacent = image[1:, :].flatten()
    elif direction == 'diagonal':
        pixels = image[:-1, :-1].flatten()
        adjacent = image[1:, 1:].flatten()
    else:
        raise ValueError("Invalid direction. Use 'horizontal', 'vertical', or 'diagonal'")
    
    return pearsonr(pixels, adjacent)[0]

def compute_npcr_uaci(enc1, enc2):
    """Correct NPCR and UACI calculation for both grayscale and RGB images"""
    if len(enc1.shape) == 3 and len(enc2.shape) == 3:
        # RGB image - compute for each channel and average
        npcr_vals, uaci_vals = [], []
        for c in range(3):
            n, u = _compute_npcr_uaci_2d(enc1[:, :, c], enc2[:, :, c])
            npcr_vals.append(n)
            uaci_vals.append(u)
        return np.mean(npcr_vals), np.mean(uaci_vals)
    else:
        return _compute_npcr_uaci_2d(enc1, enc2)

def _compute_npcr_uaci_2d(enc1, enc2):
    """Correct NPCR and UACI calculation for 2D images"""
    assert enc1.shape == enc2.shape
    enc1_flat = enc1.astype(np.float32).flatten()
    enc2_flat = enc2.astype(np.float32).flatten()
    
    diff = enc1_flat != enc2_flat
    npcr = np.mean(diff) * 100
    
    abs_diff = np.abs(enc1_flat - enc2_flat)
    uaci = np.mean(abs_diff / 255) * 100
    
    return npcr, uaci

def plot_correlations(original, encrypted, save_path):
    """Plot correlation scatter plots for original and encrypted images"""
    if len(original.shape) == 3 or len(encrypted.shape) == 3:
        # For RGB images, plot correlations for the first channel only
        original = original[:, :, 0] if len(original.shape) == 3 else original
        encrypted = encrypted[:, :, 0] if len(encrypted.shape) == 3 else encrypted
    
    directions = ['horizontal', 'vertical', 'diagonal']
    plt.figure(figsize=(15, 10))
    
    for i, direction in enumerate(directions):
        # Original image
        plt.subplot(3, 2, i*2+1)
        if direction == 'horizontal':
            x = original[:, :-1].flatten()
            y = original[:, 1:].flatten()
        elif direction == 'vertical':
            x = original[:-1, :].flatten()
            y = original[1:, :].flatten()
        elif direction == 'diagonal':
            x = original[:-1, :-1].flatten()
            y = original[1:, 1:].flatten()
        
        # Random sample for visualization
        idx = np.random.choice(len(x), min(5000, len(x)), replace=False)
        plt.scatter(x[idx], y[idx], s=1, alpha=0.5, color='blue')
        plt.title(f'Original {direction.capitalize()} Correlation')
        plt.xlabel('Pixel (x)')
        plt.ylabel('Adjacent Pixel (y)')
        
        # Encrypted image
        plt.subplot(3, 2, i*2+2)
        if direction == 'horizontal':
            x = encrypted[:, :-1].flatten()
            y = encrypted[:, 1:].flatten()
        elif direction == 'vertical':
            x = encrypted[:-1, :].flatten()
            y = encrypted[1:, :].flatten()
        elif direction == 'diagonal':
            x = encrypted[:-1, :-1].flatten()
            y = encrypted[1:, 1:].flatten()
        
        idx = np.random.choice(len(x), min(5000, len(x)), replace=False)
        plt.scatter(x[idx], y[idx], s=1, alpha=0.5, color='red')
        plt.title(f'Encrypted {direction.capitalize()} Correlation')
        plt.xlabel('Pixel (x)')
        plt.ylabel('Adjacent Pixel (y)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def add_gaussian_noise(image, variance):
    """Add Gaussian noise to image (supports RGB)"""
    if len(image.shape) == 3:  # RGB image
        noisy_image = np.zeros_like(image, dtype=np.float32)
        for c in range(3):
            channel = image[:, :, c].astype(np.float32)
            noise = np.random.normal(0, variance**0.5, channel.shape)
            noisy_channel = np.clip(channel + noise, 0, 255)
            noisy_image[:, :, c] = noisy_channel
        return noisy_image.astype(np.uint8)
    else:  # Grayscale
        noise = np.random.normal(0, variance**0.5, image.shape)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy_image.astype(np.uint8)

def add_salt_pepper_noise(image, density):
    """Add salt and pepper noise to image (supports RGB)"""
    if len(image.shape) == 3:  # RGB image
        noisy = image.copy()
        for c in range(3):
            mask = np.random.random(image.shape[:2])
            salt_mask = mask < density/2
            pepper_mask = mask > 1 - density/2
            noisy[:, :, c][salt_mask] = 255
            noisy[:, :, c][pepper_mask] = 0
        return noisy
    else:  # Grayscale
        noisy = image.copy()
        mask = np.random.random(image.shape)
        salt_mask = mask < density/2
        pepper_mask = mask > 1 - density/2
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
        return noisy

def plot_noise_attack(cipher, encrypted, original, noise_type, densities, save_path):
    """Plot noise attack analysis for either 'salt_pepper' or 'gaussian'"""
    rows = 3
    cols = len(densities) + 1
    plt.figure(figsize=(cols*3, rows*3))
    
    # Original cipher and decrypted
    plt.subplot(rows, cols, 1)
    if len(encrypted.shape) == 3:
        plt.imshow(cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(encrypted, cmap='gray')
    plt.title('Original Cipher')
    plt.axis('off')
    
    # Decrypt original cipher
    orig_decrypted = cipher.decrypt(
        os.path.join(OUTPUT_DIR, "encrypted.png"), 
        rounds=ROUNDS,
        save=False  # Return array without saving
    )
    plt.subplot(rows, cols, cols + 1)
    if len(orig_decrypted.shape) == 3:
        plt.imshow(cv2.cvtColor(orig_decrypted, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(orig_decrypted, cmap='gray')
    plt.title('Decrypted Image')
    plt.axis('off')
    
    # Add PSNR for clean decrypted
    psnr_clean = psnr(original, orig_decrypted, data_range=255)
    plt.subplot(rows, cols, 2*cols + 1)
    plt.text(0.5, 0.5, f'PSNR: {psnr_clean:.2f} dB', 
             ha='center', va='center', fontsize=12)
    plt.axis('off')
    
    # Process each density
    for i, density in enumerate(densities):
        col = i + 2  # Start from second column
        
        # Add noise to cipher
        if noise_type == 'salt_pepper':
            noisy_cipher = add_salt_pepper_noise(encrypted, density)
        else:  # Gaussian
            noisy_cipher = add_gaussian_noise(encrypted, density)
        
        # Save noisy cipher temporarily
        temp_path = os.path.join(OUTPUT_DIR, f"temp_{noise_type}_{density}.png")
        if len(noisy_cipher.shape) == 3:
            cv2.imwrite(temp_path, cv2.cvtColor(noisy_cipher, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(temp_path, noisy_cipher)
        
        # Plot noisy cipher
        plt.subplot(rows, cols, col)
        if len(noisy_cipher.shape) == 3:
            plt.imshow(cv2.cvtColor(noisy_cipher, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(noisy_cipher, cmap='gray')
        plt.title(f'{noise_type.capitalize()} {density}')
        plt.axis('off')
        
        # Decrypt the noisy cipher
        decrypted_noisy = cipher.decrypt(
            temp_path, 
            rounds=ROUNDS,
            save=False  # Return array without saving
        )
        
        # Plot decrypted image
        plt.subplot(rows, cols, col + cols)
        if len(decrypted_noisy.shape) == 3:
            plt.imshow(cv2.cvtColor(decrypted_noisy, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(decrypted_noisy, cmap='gray')
        plt.axis('off')
        
        # Calculate and show PSNR
        psnr_val = psnr(original, decrypted_noisy, data_range=255)
        plt.subplot(rows, cols, col + 2*cols)
        plt.text(0.5, 0.5, f'PSNR: {psnr_val:.2f} dB', 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        
        # Clean up temporary file
        os.remove(temp_path)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize cipher
    cipher = EnhancedDNAImageCipher("config_enhanced.json")
    
    # Get user input for file paths
    original_path = "images\lena.png"  # Can be grayscale or color
    encrypted_path = os.path.join(OUTPUT_DIR, "encrypted.png")
    decrypted_path = os.path.join(OUTPUT_DIR, "decrypted.png")
    avalanche_mod_path = os.path.join(OUTPUT_DIR, "avalanche_mod.png")
    
    # Encrypt the image if not already encrypted
    if not os.path.exists(encrypted_path):
        print(f"\nEncrypting image with {ROUNDS} rounds...")
        cipher.encrypt(
            original_path, 
            encrypted_path, 
            rounds=ROUNDS
        )
    
    # Decrypt the image if not already decrypted
    if not os.path.exists(decrypted_path):
        print(f"\nDecrypting image with {ROUNDS} rounds...")
        cipher.decrypt(
            encrypted_path, 
            decrypted_path, 
            rounds=ROUNDS
        )
    
    # Load images in their original color format
    original = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
    encrypted = cv2.imread(encrypted_path, cv2.IMREAD_UNCHANGED)
    decrypted = cv2.imread(decrypted_path, cv2.IMREAD_UNCHANGED)
    
    # Verify images loaded correctly
    for img, name in zip([original, encrypted, decrypted], 
                         ['Original', 'Encrypted', 'Decrypted']):
        if img is None:
            raise FileNotFoundError(f"{name} image not found at specified path")
    
    # Convert to RGB for consistent processing
    if len(original.shape) == 3 and original.shape[2] == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)
        decrypted = cv2.cvtColor(decrypted, cv2.COLOR_BGR2RGB)
    
    # 1. Calculate entropy
    orig_entropy = calculate_entropy(original)
    enc_entropy = calculate_entropy(encrypted)
    print(f"\nEntropy - Original: {orig_entropy:.4f}, Encrypted: {enc_entropy:.4f}")
    
    # 2. Plot histograms
    plot_histograms(original, encrypted, os.path.join(OUTPUT_DIR, 'histograms.png'))
    print(f"Histogram plot saved to '{OUTPUT_DIR}/histograms.png'")
    
    # 3. Calculate correlation coefficients
    directions = ['horizontal', 'vertical', 'diagonal']
    for direction in directions:
        orig_corr = calculate_correlation(original, direction)
        enc_corr = calculate_correlation(encrypted, direction)
        print(f"{direction.capitalize()} Correlation - Original: {orig_corr:.6f}, Encrypted: {enc_corr:.6f}")
    
    # 4. Plot correlations
    plot_correlations(original, encrypted, os.path.join(OUTPUT_DIR, 'correlations.png'))
    print(f"Correlation plots saved to '{OUTPUT_DIR}/correlations.png'")
    
    # 5. Perform avalanche test if requested
    if PERFORM_AVALANCHE_TEST:
        if not os.path.exists(avalanche_mod_path):
            print("\nPerforming avalanche test...")
            enc_orig, enc_mod = cipher.generate_avalanche_images(
                original_path, 
                os.path.join(OUTPUT_DIR, "avalanche"), 
                rounds=ROUNDS
            )
            # Load avalanche images in RGB format
            enc_orig = cv2.imread(os.path.join(OUTPUT_DIR, "avalanche_orig.png"), cv2.IMREAD_COLOR)
            enc_mod = cv2.imread(os.path.join(OUTPUT_DIR, "avalanche_mod.png"), cv2.IMREAD_COLOR)
            if len(enc_orig.shape) == 3:
                enc_orig = cv2.cvtColor(enc_orig, cv2.COLOR_BGR2RGB)
                enc_mod = cv2.cvtColor(enc_mod, cv2.COLOR_BGR2RGB)
            
            npcr, uaci = compute_npcr_uaci(enc_orig, enc_mod)
            print(f"NPCR (Avalanche): {npcr:.4f}%")
            print(f"UACI (Avalanche): {uaci:.4f}%")
        else:
            enc_orig = cv2.imread(os.path.join(OUTPUT_DIR, "avalanche_orig.png"), cv2.IMREAD_UNCHANGED)
            enc_mod = cv2.imread(os.path.join(OUTPUT_DIR, "avalanche_mod.png"), cv2.IMREAD_UNCHANGED)
            if len(enc_orig.shape) == 3:
                enc_orig = cv2.cvtColor(enc_orig, cv2.COLOR_BGR2RGB)
                enc_mod = cv2.cvtColor(enc_mod, cv2.COLOR_BGR2RGB)
            
            npcr, uaci = compute_npcr_uaci(enc_orig, enc_mod)
            print(f"NPCR (Avalanche): {npcr:.4f}%")
            print(f"UACI (Avalanche): {uaci:.4f}%")
        
        # Plot difference between avalanche images
        diff = np.abs(enc_orig.astype(int) - enc_mod.astype(int))
        if len(diff.shape) == 3:
            # For RGB, take max difference across channels
            diff = np.max(diff, axis=2)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(diff, cmap='hot')
        plt.title('Pixel Differences (Avalanche Effect)')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.hist(diff.flatten(), bins=50)
        plt.title('Difference Distribution')
        plt.savefig(os.path.join(OUTPUT_DIR, "avalanche_diff.png"))
        plt.close()
        
        print("Avalanche difference visualization saved")
    
    # 6. Noise attack analysis
    densities = [0.01, 0.02, 0.1, 0.2]
    
    # Salt & pepper noise attack
    plot_noise_attack(
        cipher, encrypted, original,
        'salt_pepper', densities,
        os.path.join(OUTPUT_DIR, "noise_attack_salt_pepper.png")
    )
    
    # Gaussian noise attack
    plot_noise_attack(
        cipher, encrypted, original,
        'gaussian', densities,
        os.path.join(OUTPUT_DIR, "noise_attack_gaussian.png")
    )
    
    print("Noise attack analysis plots saved")
    
    # 7. Calculate PSNR
    psnr_orig_dec = psnr(original, decrypted, data_range=255)
    print(f"PSNR between original and decrypted: {psnr_orig_dec:.2f} dB")

    # 8. MSE
    mse = np.mean((original.astype(float) - decrypted.astype(float)) ** 2)
    print(f"MSE: {mse:.4f}")
    
    # Save results to text file
    with open(os.path.join(OUTPUT_DIR, 'results.txt'), 'w') as f:
        f.write(f"Original Image Entropy: {orig_entropy:.4f}\n")
        f.write(f"Encrypted Image Entropy: {enc_entropy:.4f}\n\n")
        
        for direction in directions:
            orig_corr = calculate_correlation(original, direction)
            enc_corr = calculate_correlation(encrypted, direction)
            f.write(f"{direction.capitalize()} Correlation:\n")
            f.write(f"  Original: {orig_corr:.6f}\n")
            f.write(f"  Encrypted: {enc_corr:.6f}\n\n")
        
        if PERFORM_AVALANCHE_TEST:
            f.write(f"NPCR (Avalanche): {npcr:.4f}%\n")
            f.write(f"UACI (Avalanche): {uaci:.4f}%\n\n")
        
        f.write(f"PSNR (Original vs Decrypted): {psnr_orig_dec:.2f} dB\n")
        f.write(f"MSE (Original vs Decrypted): {mse:.4f}\n")
    
    print(f"\nAll results saved to '{OUTPUT_DIR}' directory")

if __name__ == "__main__":
    main()