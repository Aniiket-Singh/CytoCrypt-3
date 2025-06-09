import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_image(path, mode='L'):
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
    else:
        plt.show()

# 2. NPCR and UACI
def compute_npcr_uaci(orig, enc):
    orig = orig.flatten().astype(np.int32)
    enc = enc.flatten().astype(np.int32)
    # NPCR
    diff = orig != enc
    npcr = np.sum(diff) / diff.size * 100
    # UACI
    uaci = np.mean(np.abs(orig - enc) / 255) * 100
    return npcr, uaci

# 3. Correlation coefficients
def correlation_coefficients(img):
    # ensure 2D array
    h, w = img.shape
    # horizontal: pairs (i,j) and (i,j+1)
    horiz = [(img[i,j], img[i,j+1]) for i in range(h) for j in range(w-1)]
    vert = [(img[i,j], img[i+1,j]) for i in range(h-1) for j in range(w)]
    diag = [(img[i,j], img[i+1,j+1]) for i in range(h-1) for j in range(w-1)]
    def corr(pairs):
        a, b = zip(*pairs)
        return pearsonr(a, b)[0]
    return corr(horiz), corr(vert), corr(diag)

# 4. Entropy
def entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0,255), density=True)
    hist = hist[hist>0]
    return -np.sum(hist * np.log2(hist))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image encryption analysis')
    parser.add_argument('--original', required=True, help='Path to original plaintext image')
    parser.add_argument('--encrypted', required=True, help='Path to encrypted image')
    parser.add_argument('--decrypted', required=False, help='Path to decrypted image (optional)')
    parser.add_argument('--output-hist', required=False, help='Path to save histogram plot')
    args = parser.parse_args()

    orig = load_image(args.original)
    enc = load_image(args.encrypted)

    print('1. Histogram Analysis:')
    plot_histograms(orig, enc, save_path=args.output_hist)

    print('\n2. NPCR and UACI:')
    npcr, uaci = compute_npcr_uaci(orig, enc)
    print(f'NPCR: {npcr:.4f}%')
    print(f'UACI: {uaci:.4f}%')

    print('\n3. Correlation Coefficients:')
    cor_orig = correlation_coefficients(orig)
    cor_enc = correlation_coefficients(enc)
    dirs = ['Horizontal', 'Vertical', 'Diagonal']
    for d, co in zip(dirs, cor_orig):
        print(f'Original {d} correlation: {co:.6f}')
    for d, co in zip(dirs, cor_enc):
        print(f'Encrypted {d} correlation: {co:.6f}')

    print('\n4. Entropy:')
    ent_orig = entropy(orig)
    ent_enc = entropy(enc)
    print(f'Original Image Entropy: {ent_orig:.6f} bits')
    print(f'Encrypted Image Entropy: {ent_enc:.6f} bits')

    if args.decrypted:
        dec = load_image(args.decrypted)
        matches = np.array_equal(orig, dec)
        print(f"\nDecryption correctness: {'Success' if matches else 'Failure'}")
