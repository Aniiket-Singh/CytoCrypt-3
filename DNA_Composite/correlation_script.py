import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Constants for scatter plot
DOT_SIZE = 1  # dot size
ALPHA = 0.5  # transparency
COLORS = {'original': 'blue', 'encrypted': 'red'}

# Neighbor extraction functions
def get_neighbors(arr, direction):
    if direction == 'horizontal':
        return arr[:, :-1].flatten(), arr[:, 1:].flatten()
    if direction == 'vertical':
        return arr[:-1, :].flatten(), arr[1:, :].flatten()
    if direction == 'diagonal':
        return arr[:-1, :-1].flatten(), arr[1:, 1:].flatten()
    raise ValueError(f"Unknown direction: {direction}")

# Core plotting logic for grayscale
def plot_gray_correlations(orig, enc, save_path):
    directions = ['horizontal', 'vertical', 'diagonal']
    plt.figure(figsize=(15, 10))
    for i, direction in enumerate(directions):
        # Original subplot
        plt.subplot(3, 2, 2*i + 1)
        x_o, y_o = get_neighbors(orig, direction)
        idx_o = np.random.choice(len(x_o), min(5000, len(x_o)), replace=False)
        plt.scatter(x_o[idx_o], y_o[idx_o], s=DOT_SIZE, alpha=ALPHA, color=COLORS['original'], marker='o')
        plt.title(f'Original {direction.capitalize()} Correlation')
        plt.xlabel('p(x,y)')
        plt.ylabel('p(x'+('+'+ '1,y' if direction=='horizontal' else ',y+1)' if direction=='vertical' else '+1,y+1)'))
        
        # Encrypted subplot
        plt.subplot(3, 2, 2*i + 2)
        x_e, y_e = get_neighbors(enc, direction)
        idx_e = np.random.choice(len(x_e), min(5000, len(x_e)), replace=False)
        plt.scatter(x_e[idx_e], y_e[idx_e], s=DOT_SIZE, alpha=ALPHA, color=COLORS['encrypted'], marker='o')
        plt.title(f'Encrypted {direction.capitalize()} Correlation')
        plt.xlabel('p(x,y)')
        plt.ylabel('p(x'+('+'+ '1,y' if direction=='horizontal' else ',y+1)' if direction=='vertical' else '+1,y+1)'))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined grayscale correlations to '{save_path}'")

# Core plotting logic for RGB channels
def plot_rgb_correlations(orig, enc, save_path):
    # Create a separate figure per channel
    for ch, idx in zip(['R', 'G', 'B'], range(3)):
        plt.figure(figsize=(15, 10))
        for i, direction in enumerate(['horizontal', 'vertical', 'diagonal']):
            # Original
            plt.subplot(3, 2, 2*i + 1)
            ochan = orig[:, :, idx]
            x_o, y_o = get_neighbors(ochan, direction)
            idx_o = np.random.choice(len(x_o), min(5000, len(x_o)), replace=False)
            plt.scatter(x_o[idx_o], y_o[idx_o], s=DOT_SIZE, alpha=ALPHA, color=COLORS['original'], marker='o')
            plt.title(f'{ch} Original {direction.capitalize()}')
            plt.xlabel('p(x,y)')
            plt.ylabel('p(x'+('+'+ '1,y' if direction=='horizontal' else ',y+1)' if direction=='vertical' else '+1,y+1)'))

            # Encrypted
            plt.subplot(3, 2, 2*i + 2)
            echan = enc[:, :, idx]
            x_e, y_e = get_neighbors(echan, direction)
            idx_e = np.random.choice(len(x_e), min(5000, len(x_e)), replace=False)
            plt.scatter(x_e[idx_e], y_e[idx_e], s=DOT_SIZE, alpha=ALPHA, color=COLORS['encrypted'], marker='o')
            plt.title(f'{ch} Encrypted {direction.capitalize()}')
            plt.xlabel('p(x,y)')
            plt.ylabel('p(x'+('+'+ '1,y' if direction=='horizontal' else ',y+1)' if direction=='vertical' else '+1,y+1)'))
        plt.tight_layout()
        fname = save_path.replace('.png', f'_{ch}.png')
        plt.savefig(fname)
        plt.close()
        print(f"Saved {ch}-channel correlations to '{fname}'")

# Utility functions

def load_image(path):
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr

# Main entry
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python correlation_script.py <original.png> <encrypted.png>")
        sys.exit(1)
    orig_path, enc_path = sys.argv[1], sys.argv[2]
    if not os.path.isfile(orig_path) or not os.path.isfile(enc_path):
        print("One or both files not found.")
        sys.exit(1)

    orig = load_image(orig_path)
    enc = load_image(enc_path)

    # Decide grayscale vs RGB
    if orig.ndim == 2 or (orig.ndim == 3 and orig.shape[2] == 3 and np.all(orig[:,:,0]==orig[:,:,1])):
        # Grayscale case
        ogray = orig if orig.ndim == 2 else orig[:, :, 0]
        egray = enc if enc.ndim == 2 else enc[:, :, 0]
        plot_gray_correlations(ogray, egray, 'correlations_gray.png')
    else:
        # RGB case
        plot_rgb_correlations(orig, enc, 'correlations_rgb.png')

    print("Done.")
