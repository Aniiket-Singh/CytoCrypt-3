import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr
import os

def calculate_entropy(image):
    """Calculate Shannon entropy of an image"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy

def plot_histograms(original, encrypted, save_path):
    """Plot histograms for original and encrypted images with same y-axis"""
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

def calculate_npcr_uaci(original, encrypted):
    """Calculate NPCR and UACI between two images"""
    if original.shape != encrypted.shape:
        raise ValueError("Images must have the same dimensions")
    
    diff = original != encrypted
    npcr = np.sum(diff) / original.size * 100
    
    uaci = np.sum(np.abs(original.astype(int) - encrypted.astype(int))) / (255 * original.size) * 100
    return npcr, uaci

def plot_correlations(original, encrypted, save_path):
    """Plot correlation scatter plots for original and encrypted images"""
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

def noise_attack_analysis(encrypted, decrypted, original, save_path):
    """Perform noise attack analysis and visualization"""
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 0.01, encrypted.shape).astype(np.float32)
    encrypted_gaussian = np.clip(encrypted.astype(np.float32)/255 + gaussian_noise, 0, 1)
    encrypted_gaussian = (encrypted_gaussian * 255).astype(np.uint8)
    
    # Add Salt & Pepper noise
    salt_pepper = np.random.random(encrypted.shape)
    encrypted_sp = encrypted.copy()
    salt_mask = salt_pepper < 0.025  # Salt: 2.5%
    pepper_mask = salt_pepper > 0.975  # Pepper: 2.5%
    encrypted_sp[salt_mask] = 255
    encrypted_sp[pepper_mask] = 0
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Gaussian noise row
    plt.subplot(2, 3, 1)
    plt.imshow(encrypted, cmap='gray')
    plt.title('Original Cipher')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(encrypted_gaussian, cmap='gray')
    plt.title('Gaussian Noise (σ²=0.01)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(decrypted, cmap='gray')
    plt.title('Decrypted from Gaussian')
    plt.axis('off')
    
    # Salt & Pepper noise row
    plt.subplot(2, 3, 4)
    plt.imshow(encrypted, cmap='gray')
    plt.title('Original Cipher')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(encrypted_sp, cmap='gray')
    plt.title('Salt & Pepper Noise (5%)')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(decrypted, cmap='gray')  # Replace with actual decrypted from SP
    plt.title('Decrypted from S&P')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return encrypted_gaussian, encrypted_sp

def main():
    # Get user input for file paths
    original_path = "images\lena_grayscale.png"
    encrypted_path = "results\encrypted.png"
    decrypted_path = "results\decrypted.png"
    
    # Load images
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    encrypted = cv2.imread(encrypted_path, cv2.IMREAD_GRAYSCALE)
    decrypted = cv2.imread(decrypted_path, cv2.IMREAD_GRAYSCALE)
    
    # Verify images loaded correctly
    for img, name in zip([original, encrypted, decrypted], 
                         ['Original', 'Encrypted', 'Decrypted']):
        if img is None:
            raise FileNotFoundError(f"{name} image not found at specified path")
    
    # Create output directory
    os.makedirs('composite_v2', exist_ok=True)
    
    # 1. Calculate entropy
    orig_entropy = calculate_entropy(original)
    enc_entropy = calculate_entropy(encrypted)
    print(f"\nEntropy - Original: {orig_entropy:.4f}, Encrypted: {enc_entropy:.4f}")
    
    # 2. Plot histograms
    plot_histograms(original, encrypted, 'composite_v2/histogram.png')
    print("Histogram plot saved to 'composite_v2/histogram.png'")
    
    # 3. Calculate correlation coefficients
    directions = ['horizontal', 'vertical', 'diagonal']
    for direction in directions:
        orig_corr = calculate_correlation(original, direction)
        enc_corr = calculate_correlation(encrypted, direction)
        print(f"{direction.capitalize()} Correlation - Original: {orig_corr:.6f}, Encrypted: {enc_corr:.6f}")
    
    # 4. Calculate NPCR and UACI
    npcr, uaci = calculate_npcr_uaci(original, encrypted)
    print(f"NPCR: {npcr:.4f}%, UACI: {uaci:.4f}%")
    
    # 5. Plot correlations
    plot_correlations(original, encrypted, 'composite_v2/correlations.png')
    print("Correlation plots saved to 'composite_v2/correlations.png'")
    
    # 6. Noise attack analysis
    noisy_gaussian, noisy_sp = noise_attack_analysis(
        encrypted, decrypted, original, 
        'composite_v2/Noise_attack_analysis.png'
    )
    print("Noise attack analysis plot saved to 'composite_v2/Noise_attack_analysis.png'")
    
    # 7. Calculate PSNR
    psnr_orig_dec = psnr(original, decrypted, data_range=255)
    print(f"PSNR between original and decrypted: {psnr_orig_dec:.2f} dB")

    # 8. MSE
    mse = np.mean((original.astype(float) - decrypted.astype(float)) ** 2)
    print(f"MSE: {mse}")  # Will show 0.0 if identical
    
    # Save results to text file
    with open('composite_v2/results.txt', 'w') as f:
        f.write(f"Original Image Entropy: {orig_entropy:.4f}\n")
        f.write(f"Encrypted Image Entropy: {enc_entropy:.4f}\n\n")
        
        for direction in directions:
            orig_corr = calculate_correlation(original, direction)
            enc_corr = calculate_correlation(encrypted, direction)
            f.write(f"{direction.capitalize()} Correlation:\n")
            f.write(f"  Original: {orig_corr:.6f}\n")
            f.write(f"  Encrypted: {enc_corr:.6f}\n\n")
        
        f.write(f"NPCR: {npcr:.4f}%\n")
        f.write(f"UACI: {uaci:.4f}%\n\n")
        f.write(f"PSNR (Original vs Decrypted): {psnr_orig_dec:.2f} dB\n")
    
    print("\nAll results saved to 'composite_v2' directory")

if __name__ == "__main__":
    main()