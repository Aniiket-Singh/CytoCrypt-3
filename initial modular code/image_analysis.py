import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# === User-configurable paths ===
# Enter the file paths for your images below
original_path = 'images/lena_grayscale.png'
encrypted_path = 'composite-implement_encrypted_enhanced.png'
decrypted_path = ''  # Set to None if not using
output_hist_path = 'results/lena_composite_implemented.png'         # Set to None to display instead of saving

# Create results directory if it doesn't exist
if output_hist_path and not os.path.exists(os.path.dirname(output_hist_path)):
    os.makedirs(os.path.dirname(output_hist_path))

# === Utility functions ===
def load_image(path, mode='RGB'):
    """Load an image as a NumPy array in the specified mode."""
    img = Image.open(path).convert(mode)
    return np.array(img)

def is_grayscale(img_array):
    """Check if an image is grayscale (single channel or all channels identical)"""
    if len(img_array.shape) == 2:
        return True  # Single channel image
    
    if len(img_array.shape) == 3:
        # Check if all channels are identical
        if np.all(img_array[:, :, 0] == img_array[:, :, 1]) and \
           np.all(img_array[:, :, 0] == img_array[:, :, 2]):
            return True
    
    return False

# === Histogram Functions ===
def plot_histograms(orig, enc, save_path=None):
    """Plot histograms only if both images are grayscale"""
    if not is_grayscale(orig) or not is_grayscale(enc):
        print("Skipping histogram: One or both images are not grayscale")
        return
    
    # Convert to grayscale arrays if needed
    if len(orig.shape) == 3:
        orig = orig[:, :, 0]  # Use first channel (all channels are identical)
    if len(enc.shape) == 3:
        enc = enc[:, :, 0]  # Use first channel (all channels are identical)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original image histogram
    plt.subplot(1, 2, 1)
    plt.title('Original Image Histogram')
    plt.hist(orig.flatten(), bins=256, range=(0, 255), density=True)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    
    # Encrypted image histogram
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

# === Security Metrics Functions ===
def compute_npcr_uaci(orig, enc):
    """Compute NPCR and UACI metrics"""
    # Convert to float to prevent overflow
    orig_flat = orig.astype(np.float32).flatten()
    enc_flat = enc.astype(np.float32).flatten()
    
    # NPCR (Number of Pixel Change Rate)
    diff = orig_flat != enc_flat
    npcr = np.sum(diff) / diff.size * 100
    
    # UACI (Unified Average Changing Intensity)
    abs_diff = np.abs(orig_flat - enc_flat)
    uaci = np.mean(abs_diff / 255) * 100
    
    return npcr, uaci

def correlation_coefficients(img):
    """Compute correlation coefficients in horizontal, vertical and diagonal directions"""
    # Convert to grayscale for correlation analysis
    if len(img.shape) == 3:
        # If it's a true grayscale image stored as RGB
        if is_grayscale(img):
            img = img[:, :, 0]
        else:
            # Convert color image to grayscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    h, w = img.shape
    # Horizontal neighbors
    horiz = img[:, :-1].flatten()
    horiz_neighbors = img[:, 1:].flatten()
    # Vertical neighbors
    vert = img[:-1, :].flatten()
    vert_neighbors = img[1:, :].flatten()
    # Diagonal neighbors
    diag = img[:-1, :-1].flatten()
    diag_neighbors = img[1:, 1:].flatten()
    
    # Calculate correlation coefficients
    horiz_corr = pearsonr(horiz, horiz_neighbors)[0]
    vert_corr = pearsonr(vert, vert_neighbors)[0]
    diag_corr = pearsonr(diag, diag_neighbors)[0]
    
    return horiz_corr, vert_corr, diag_corr

def entropy(img):
    """Calculate image entropy"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        if is_grayscale(img):
            img = img[:, :, 0]  # Use first channel
        else:
            # Convert color image to grayscale
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    # Calculate histogram and probabilities
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    prob = hist / hist.sum()
    prob = prob[prob > 0]  # Remove zero probabilities
    
    return -np.sum(prob * np.log2(prob))

# === Main Analysis ===
if __name__ == '__main__':
    print("Starting image analysis...")
    
    # Load images in RGB mode
    orig = load_image(original_path, 'RGB')
    enc = load_image(encrypted_path, 'RGB')
    
    # 1. Histogram Analysis (only for grayscale images)
    print('\n1. Histogram Analysis:')
    plot_histograms(orig, enc, save_path=output_hist_path)
    
    # 2. NPCR and UACI
    print('\n2. NPCR and UACI:')
    npcr, uaci = compute_npcr_uaci(orig, enc)
    print(f'NPCR: {npcr:.4f}% (should be >99.5% for good encryption)')
    print(f'UACI: {uaci:.4f}% (should be ~33.4% for good encryption)')
    
    # 3. Correlation Coefficients
    print('\n3. Correlation Coefficients:')
    cor_orig = correlation_coefficients(orig)
    cor_enc = correlation_coefficients(enc)
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    
    print("\nOriginal Image:")
    for d, c in zip(directions, cor_orig):
        print(f'  {d} correlation: {c:.6f} (should be close to 1)')
    
    print("\nEncrypted Image:")
    for d, c in zip(directions, cor_enc):
        print(f'  {d} correlation: {c:.6f} (should be close to 0)')
    
    # 4. Entropy
    print('\n4. Entropy:')
    ent_orig = entropy(orig)
    ent_enc = entropy(enc)
    print(f'Original Image Entropy: {ent_orig:.6f} bits')
    print(f'Encrypted Image Entropy: {ent_enc:.6f} bits (should be close to 8)')
    
    # 5. Decryption correctness (optional)
    if decrypted_path:
        try:
            dec = load_image(decrypted_path, 'RGB')
            match = np.array_equal(orig, dec)
            print(f"\n5. Decryption correctness: {'SUCCESS' if match else 'FAILURE'}")
            if not match:
                # Calculate mismatch percentage
                mismatch = np.mean(orig != dec) * 100
                print(f"  Mismatch percentage: {mismatch:.6f}%")
        except FileNotFoundError:
            print(f"\nDecrypted image not found at {decrypted_path}")
    
    print("\nAnalysis complete!")