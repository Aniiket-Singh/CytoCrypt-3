#!/usr/bin/env python3
"""
mod.py: Modify exactly one pixel in an input image to generate a slightly altered version,
useful for UACI/NPCR testing in encryption analysis.

Usage:
    python mod.py path/to/image.png
    python mod.py path/to/image.png -o altered.png

By default, the output is saved as `mod.png` in the current directory.
"""
import argparse
import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modify one pixel of the image for UACI/NPCR tests."
    )
    parser.add_argument(
        "input_image",
        help="Path to the input image file"
    )
    parser.add_argument(
        "-o", "--output",
        default="mod.png",
        help="Filename for the modified image (default: mod.png)"
    )
    return parser.parse_args()


def modify_one_pixel(img: Image.Image) -> Image.Image:
    """
    Alters exactly one random pixel in the image by incrementing one of its RGB channels by 1 (mod 256).
    """
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size

    # Choose a random pixel coordinate
    x = random.randrange(width)
    y = random.randrange(height)

    # Get original channel values
    r, g, b = pixels[x, y]

    # Randomly pick one channel to alter
    channel = random.choice([0, 1, 2])
    if channel == 0:
        r = (r + 1) % 256
    elif channel == 1:
        g = (g + 1) % 256
    else:
        b = (b + 1) % 256

    pixels[x, y] = (r, g, b)
    return img


def main():
    args = parse_args()

    # Load image
    try:
        img = Image.open(args.input_image)
    except Exception as e:
        print(f"Error: Couldn't open image '{args.input_image}': {e}")
        return

    # Modify one pixel
    modified = modify_one_pixel(img)

    # Save output
    try:
        modified.save(args.output)
        print(f"Modified image saved as '{args.output}'")
    except Exception as e:
        print(f"Error: Couldn't save modified image '{args.output}': {e}")


if __name__ == "__main__":
    main()
