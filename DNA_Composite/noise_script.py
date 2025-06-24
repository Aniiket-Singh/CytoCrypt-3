import argparse
import os
import sys
import numpy as np
import cv2

def add_gaussian_noise(image, var):
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
    noisy = image.astype('float32') + gauss * 255
    noisy = np.clip(noisy, 0, 255).astype('uint8')
    return noisy

def add_salt_pepper_noise(image, amount):
    noisy = image.copy()
    # amount is fraction of pixels to alter
    num_pixels = np.product(image.shape[:2])
    num_salt = np.ceil(amount * num_pixels * 0.5).astype(int)
    num_pepper = np.ceil(amount * num_pixels * 0.5).astype(int)

    # Salt noise (white pixels)
    coords = [
        np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]
    ]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise (black pixels)
    coords = [
        np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]
    ]
    noisy[coords[0], coords[1]] = 0
    return noisy

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Add Gaussian or Salt-Pepper noise to an image.'
    )
    parser.add_argument(
        'noise_type',
        choices=['gaussian', 'sp'],
        help='Type of noise to add (gaussian or sp).'
    )
    parser.add_argument(
        'intensity',
        type=float,
        help='Intensity of noise: variance for gaussian, fraction for sp (e.g., 0.05 for 5%%).'
    )
    parser.add_argument(
        'image_path',
        help='Path to the input image file.'
    )
    parser.add_argument(
        '--output',
        help='Path to save the noisy image. If not provided, a default name will be used.',
        default=None
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not os.path.isfile(args.image_path):
        print(f"Error: File '{args.image_path}' not found.")
        sys.exit(1)

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Unable to read image '{args.image_path}'.")
        sys.exit(1)

    if args.noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(image, args.intensity)
        suffix = f"_gaussian_{args.intensity}"
    else:
        noisy_img = add_salt_pepper_noise(image, args.intensity)
        suffix = f"_sp_{args.intensity}"

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image_path)
        output_path = f"{base}{suffix}{ext}"

    # Save the noisy image
    cv2.imwrite(output_path, noisy_img)
    print(f"Noisy image saved to: {output_path}")

if __name__ == '__main__':
    main()
