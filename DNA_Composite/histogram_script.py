import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def compute_histograms(image_path, bins=256, range=(0, 255)):
    """
    Load an image and compute histograms for each channel.

    Returns:
        histograms: List of hist arrays for each channel.
    """
    img = Image.open(image_path)
    img_arr = np.array(img)
    if img_arr.ndim == 2:
        hist, _ = np.histogram(img_arr.flatten(), bins=bins, range=range)
        return [hist]
    elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
        histograms = []
        for ch in range(3):
            channel_data = img_arr[:, :, ch].flatten()
            hist, _ = np.histogram(channel_data, bins=bins, range=range)
            histograms.append(hist)
        return histograms
    else:
        raise ValueError(f"Unsupported image format: {img_arr.shape}")


def plot_side_by_side_histograms(enc_hists, orig_hists, output_path="histogram.png"):
    """
    Plot histograms side by side for each channel and save the figure.
    Left column: Original (decrypted), right column: Encrypted.
    """
    # Determine channel labels and colors
    if len(orig_hists) == 3:
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
    else:
        channels = ['Gray']
        colors = ['gray']

    n = len(channels)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 4 * n))

    for i, (ch_name, color) in enumerate(zip(channels, colors)):
        # Select axes: [row, 0] original, [row, 1] encrypted
        if n > 1:
            ax_orig = axes[i, 0]
            ax_enc = axes[i, 1]
        else:
            ax_orig = axes[0]
            ax_enc = axes[1]

        x = np.arange(len(orig_hists[i]))

        # Plot filled histograms
        ax_orig.fill_between(x, orig_hists[i], step='mid', alpha=0.5, color=color)
        ax_orig.set_title(f"Original {ch_name} Histogram")
        ax_orig.set_xlabel("Pixel Value")
        ax_orig.set_ylabel("Frequency")

        ax_enc.fill_between(x, enc_hists[i], step='mid', alpha=0.5, color=color)
        ax_enc.set_title(f"Encrypted {ch_name} Histogram")
        ax_enc.set_xlabel("Pixel Value")
        ax_enc.set_ylabel("Frequency")

        # Sync y-axis
        max_freq = max(orig_hists[i].max(), enc_hists[i].max())
        ax_orig.set_ylim(0, max_freq)
        ax_enc.set_ylim(0, max_freq)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Histogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot histograms of encrypted and decrypted images side by side.")
    parser.add_argument('encrypted_image', help="Path to the encrypted image file.")
    parser.add_argument('decrypted_image', help="Path to the decrypted (original) image file.")
    parser.add_argument('--output', default='histogram.png', help="Output filename for the histogram plot.")
    args = parser.parse_args()

    enc_hists = compute_histograms(args.encrypted_image)
    orig_hists = compute_histograms(args.decrypted_image)

    if len(enc_hists) != len(orig_hists):
        raise ValueError("Encrypted and original images have different channel counts.")

    plot_side_by_side_histograms(enc_hists, orig_hists, args.output)


if __name__ == "__main__":
    main()
