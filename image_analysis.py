# image_analysis.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# === User-configurable paths ===
# Enter the file paths for your images below
original_path = 'images\lena.png'
encrypted_path = 'encrypted.png'
decrypted_path = ''  # Set to None if not using
output_hist_path = 'results\lena_result_with_r1_rgb.png'         # Set to None to display instead of saving

# === Utility functions ===
def load_image(path, mode='L'):
    """Load an image as a NumPy array in the specified mode."""
    img = Image.open(path).convert(mode)
    return np.array(img)

# 1. Histogram plotting
def plot_histograms(orig, enc, save_path=None):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title('Original Image Histogram')
    plt.hist(orig.flatten(), bins=256, range=(0,255), density=True)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')

    plt.subplot(1,2,2)
    plt.title('Encrypted Image Histogram')
    plt.hist(enc.flatten(), bins=256, range=(0,255), density=True)
    plt.xlabel('Pixel value')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Histogram plot saved to {save_path}")
    else:
        plt.show()

# 2. NPCR and UACI
def compute_npcr_uaci(orig, enc):
    orig_flat = orig.flatten().astype(np.int32)
    enc_flat = enc.flatten().astype(np.int32)
    # NPCR
    diff = orig_flat != enc_flat
    npcr = np.sum(diff) / diff.size * 100
    # UACI
    uaci = np.mean(np.abs(orig_flat - enc_flat) / 255) * 100
    return npcr, uaci

# 3. Correlation coefficients
def correlation_coefficients(img):
    h, w = img.shape
    horiz = [(img[i,j], img[i,j+1]) for i in range(h) for j in range(w-1)]
    vert  = [(img[i,j], img[i+1,j]) for i in range(h-1) for j in range(w)]
    diag  = [(img[i,j], img[i+1,j+1]) for i in range(h-1) for j in range(w-1)]
    def corr(pairs):
        a, b = zip(*pairs)
        return pearsonr(a, b)[0]
    return corr(horiz), corr(vert), corr(diag)

# 4. Entropy
def entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,255), density=True)
    hist = hist[hist>0]
    return -np.sum(hist * np.log2(hist))

# === Main Analysis ===
if __name__ == '__main__':
    # Load images
    orig = load_image(original_path)
    enc = load_image(encrypted_path)

    # 1. Histogram Analysis
    print('1. Histogram Analysis:')
    plot_histograms(orig, enc, save_path=output_hist_path)

    # 2. NPCR and UACI
    print('\n2. NPCR and UACI:')
    npcr, uaci = compute_npcr_uaci(orig, enc)
    print(f'NPCR: {npcr:.4f}%')
    print(f'UACI: {uaci:.4f}%')

    # 3. Correlation Coefficients
    print('\n3. Correlation Coefficients:')
    cor_orig = correlation_coefficients(orig)
    cor_enc  = correlation_coefficients(enc)
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    for d, c in zip(directions, cor_orig):
        print(f'Original {d} correlation: {c:.6f}')
    for d, c in zip(directions, cor_enc):
        print(f'Encrypted {d} correlation: {c:.6f}')

    # 4. Entropy
    print('\n4. Entropy:')
    ent_orig = entropy(orig)
    ent_enc  = entropy(enc)
    print(f'Original Image Entropy: {ent_orig:.6f} bits')
    print(f'Encrypted Image Entropy: {ent_enc:.6f} bits')

    # 5. Decryption correctness (optional)
    if decrypted_path:
        try:
            dec = load_image(decrypted_path)
            match = np.array_equal(orig, dec)
            print(f"\nDecryption correctness: {'Success' if match else 'Failure'}")
        except FileNotFoundError:
            print(f"\nDecrypted image not found at {decrypted_path}")
