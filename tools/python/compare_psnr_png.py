#!/usr/bin/env python3
"""
Compare two PNG images and calculate MSE, SNR, and PSNR metrics.
"""

import argparse
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def calculate_psnr(img1, img2, max_pixel_value=255.0):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def calculate_snr(img1, img2):
    """Calculate Signal-to-Noise Ratio between two images."""
    signal_power = np.mean(img1.astype(float) ** 2)
    noise_power = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    if noise_power == 0:
        return float('inf')
    if signal_power == 0:
        return 0.0
    
    return 10 * np.log10(signal_power / noise_power)


def load_image(filepath):
    """Load an image and convert to numpy array."""
    try:
        img = Image.open(filepath)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_difference_image(img1, img2, output_path, colormap='hot'):
    """
    Generate a difference image showing pixel-wise differences.
    Similar to FLIP application output.
    
    Args:
        img1: First image (reference) as numpy array
        img2: Second image (comparison) as numpy array
        output_path: Path to save the difference image
        colormap: Matplotlib colormap to use (default: 'hot')
    """
    # Convert to float for calculations
    img1_float = img1.astype(float)
    img2_float = img2.astype(float)
    
    # Handle multi-channel images (RGB/RGBA)
    if len(img1.shape) == 3:
        # Calculate per-pixel error (L2 norm across channels)
        diff = np.sqrt(np.sum((img1_float - img2_float) ** 2, axis=2))
    else:
        # Grayscale image
        diff = np.abs(img1_float - img2_float)
    
    # Normalize to 0-1 range for visualization
    # Use max possible difference for normalization
    if len(img1.shape) == 3:
        max_diff = 255.0 * np.sqrt(img1.shape[2])  # Max L2 distance
    else:
        max_diff = 255.0
    
    diff_normalized = np.clip(diff / max_diff, 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    diff_colored = cmap(diff_normalized)
    
    # Convert to 8-bit RGB
    diff_rgb = (diff_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save the difference image
    diff_image = Image.fromarray(diff_rgb)
    diff_image.save(output_path)
    print(f"Difference image saved to: {output_path}")
    
    # Print statistics about the differences
    print(f"Difference statistics:")
    print(f"  Min error: {np.min(diff):.4f}")
    print(f"  Max error: {np.max(diff):.4f}")
    print(f"  Mean error: {np.mean(diff):.4f}")
    print(f"  Median error: {np.median(diff):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate MSE, SNR, and PSNR metrics between two PNG images'
    )
    parser.add_argument('image1', help='Path to the first PNG image (reference)')
    parser.add_argument('image2', help='Path to the second PNG image (comparison)')
    parser.add_argument('-o', '--output', '--diff-output', 
                        dest='diff_output',
                        help='Output path for difference image (FLIP-like visualization)')
    parser.add_argument('--colormap', 
                        default='hot',
                        choices=['hot', 'viridis', 'jet', 'plasma', 'inferno', 'magma', 'turbo'],
                        help='Colormap for difference visualization (default: hot)')
    
    args = parser.parse_args()
    
    # Load images
    print(f"Loading images...")
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)
    
    # Check if images have the same shape
    if img1.shape != img2.shape:
        print(f"Error: Images have different dimensions", file=sys.stderr)
        print(f"  Image 1: {img1.shape}", file=sys.stderr)
        print(f"  Image 2: {img2.shape}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate metrics
    mse = calculate_mse(img1, img2)
    psnr = calculate_psnr(img1, img2)
    snr = calculate_snr(img1, img2)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Image Comparison Metrics")
    print(f"{'='*50}")
    print(f"Image 1: {args.image1}")
    print(f"Image 2: {args.image2}")
    print(f"Image shape: {img1.shape}")
    print(f"{'-'*50}")
    print(f"MSE  (Mean Squared Error):       {mse:.4f}")
    print(f"SNR  (Signal-to-Noise Ratio):    {snr:.4f} dB")
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.4f} dB")
    print(f"{'='*50}\n")
    
    # Generate difference image if requested
    if args.diff_output:
        print(f"Generating difference image...")
        generate_difference_image(img1, img2, args.diff_output, args.colormap)
        print()


if __name__ == '__main__':
    main()

