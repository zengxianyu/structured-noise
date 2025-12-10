#!/usr/bin/env python3
"""
PyTorch Structured Noise Generator with Frequency Soft Cutoff

This module provides functions to generate structured noise by combining
Gaussian noise magnitude with image phase using frequency soft cutoff.
"""

import torch


def create_frequency_soft_cutoff_mask(height: int, width: int, cutoff_radius: float, 
                                    transition_width: float = 5.0, device: torch.device = None) -> torch.Tensor:
    """
    Create a smooth frequency cutoff mask for low-pass filtering.
    
    Args:
        height: Image height
        width: Image width  
        cutoff_radius: Frequency cutoff radius (0 = no structure, max_radius = full structure)
        transition_width: Width of smooth transition (smaller = sharper cutoff)
        device: Device to create tensor on
    
    Returns:
        torch.Tensor: Frequency mask of shape (height, width)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Create frequency coordinates
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing='ij')
    
    # Calculate distance from center
    center_u, center_v = height // 2, width // 2
    frequency_radius = torch.sqrt((u - center_u)**2 + (v - center_v)**2)
    
    # Create smooth transition mask
    mask = torch.exp(-(frequency_radius - cutoff_radius)**2 / (2 * transition_width**2))
    mask = torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)
    
    return mask

def clip_frequency_magnitude(noise_magnitudes, clip_percentile=0.95):
    """Clip frequency domain magnitude to prevent large values."""
    
    # Calculate clipping threshold
    clip_threshold = torch.quantile(noise_magnitudes, clip_percentile)
    
    # Clip large values
    clipped_magnitudes = torch.clamp(noise_magnitudes, max=clip_threshold)
    
    return clipped_magnitudes

def generate_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    input_noise : torch.Tensor = None,
    sampling_method: str = 'fft',

) -> torch.Tensor:
    """
    Generate structured noise for a batch of images using frequency soft cutoff.
    Reduces boundary artifacts by padding images before FFT processing.
    
    Args:
        image_batch: Batch of image tensors of shape (B, C, H, W)
        noise_std: Standard deviation for Gaussian noise
        pad_factor: Padding factor (1.5 = 50% padding, 2.0 = 100% padding)
        cutoff_radius: Frequency cutoff radius (None = auto-calculate)
        transition_width: Width of smooth transition for frequency cutoff
        input_noise: Optional input noise tensor to use instead of generating new noise.
        sampling_method: Method to sample noise magnitude ('fft', 'cdf', 'two-gaussian')
    
    Returns:
        torch.Tensor: Batch of structured noise tensors of shape (B, C, H, W)
    """
    assert sampling_method in ['fft', 'cdf', 'two-gaussian']
    # Ensure tensor is on the correct device
    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch = image_batch.float()
    
    # Calculate padding size for overlap-add method
    pad_h = int(height * (pad_factor - 1))
    pad_h = pad_h // 2 * 2 # make it even
    pad_w = int(width * (pad_factor - 1))
    pad_w = pad_w // 2 * 2 # make it even
    
    # Pad images with reflection to reduce boundary artifacts
    padded_images = torch.nn.functional.pad(
        image_batch, 
        (pad_w//2, pad_w//2, pad_h//2, pad_h//2), 
        mode='reflect'  # Mirror edges for natural transitions
    )
    
    # Calculate padded dimensions
    padded_height = height + pad_h
    padded_width = width + pad_w
    
    # Create frequency soft cutoff mask only if cutoff_radius is provided
    if cutoff_radius is not None:
        cutoff_radius = min(min(padded_height/2, padded_width/2), cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            padded_height, padded_width, cutoff_radius, transition_width, device
        )
    else:
        # No cutoff - preserve all frequencies (full structure preservation)
        freq_mask = torch.ones(padded_height, padded_width, device=device)
    
    # Apply 2D FFT to padded images
    fft = torch.fft.fft2(padded_images, dim=(-2, -1))
    
    # Shift zero frequency to center
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    
    # Extract phase and magnitude for all images
    image_phases = torch.angle(fft_shifted)
    image_phases = clip_frequency_magnitude(image_phases)
    image_magnitudes = torch.abs(fft_shifted)
    
    if input_noise is not None:
        # Use provided noise
        noise_batch = torch.nn.functional.pad(
            input_noise, 
            (pad_w//2, pad_w//2, pad_h//2, pad_h//2), 
            mode='reflect'  # Mirror edges for natural transitions
        )
        noise_batch = noise_batch.float()
    else:
        # Generate Gaussian noise for the padded size
        noise_batch = torch.randn_like(padded_images)
    
    # Extract noise magnitude and phase
    if sampling_method == 'fft':
        # Apply 2D FFT to noise batch
        noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
        noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))
    
        noise_magnitudes = torch.abs(noise_fft_shifted)
        noise_phases = torch.angle(noise_fft_shifted)
    elif sampling_method == 'cdf':
        # The magnitude of FFT of Gaussian noise follows a Rayleigh distribution.
        # We can sample it directly.
        # The scale of the Rayleigh distribution is related to the std of the Gaussian noise
        # and the size of the FFT.
        # For an N-point FFT of Gaussian noise with variance sigma^2, the variance of
        # the real and imaginary parts of the FFT coefficients is N*sigma^2.
        # The scale parameter for the Rayleigh distribution is sqrt(N*sigma^2 / 2).
        # Here, N = padded_height * padded_width.

        N = padded_height * padded_width
        rayleigh_scale = (N / 2)**0.5
        
        ## Sample from a standard Rayleigh distribution (scale=1) and then scale it.
        uu = torch.rand(size=image_magnitudes.shape, device=device)
        noise_magnitudes = rayleigh_scale * torch.sqrt(-2.0 * torch.log(uu))
        if input_noise is not None:
            noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
            noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))
        
            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi - torch.pi
    elif sampling_method == 'two-gaussian':
        N = padded_height * padded_width
        rayleigh_scale = (N / 2)**0.5
        # A standard Rayleigh can be generated from two standard normal distributions.
        u1 = torch.randn_like(image_magnitudes)
        u2 = torch.randn_like(image_magnitudes)
        noise_magnitudes = rayleigh_scale * torch.sqrt(u1**2 + u2**2)
        if input_noise is not None:
            noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
            noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))
        
            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi - torch.pi
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    noise_magnitudes = clip_frequency_magnitude(noise_magnitudes)
    
    # Scale noise magnitude by standard deviation
    noise_magnitudes = noise_magnitudes * noise_std
    
    # Apply frequency soft cutoff to mix phases
    # Low frequencies (within cutoff) use image phase, high frequencies use noise phase
    mixed_phases = freq_mask.unsqueeze(0).unsqueeze(0) * image_phases + \
                   (1 - freq_mask.unsqueeze(0).unsqueeze(0)) * noise_phases
    
    # Combine magnitude and mixed phase for all images
    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)
    # Shift zero frequency back to corner
    fft_unshifted = torch.fft.ifftshift(fft_combined, dim=(-2, -1))
    # Apply inverse FFT
    structured_noise_padded = torch.fft.ifft2(fft_unshifted, dim=(-2, -1))
    # Take real part
    structured_noise_padded = torch.real(structured_noise_padded)

    clamp_mask = (structured_noise_padded < -5) + (structured_noise_padded > 5)
    clamp_mask = (clamp_mask > 0).float()

    structured_noise_padded = structured_noise_padded * (1 - clamp_mask) + noise_batch * clamp_mask
    
    # Crop back to original size (remove padding)
    structured_noise_batch = structured_noise_padded[
        :, :, 
        pad_h//2:pad_h//2 + height, 
        pad_w//2:pad_w//2 + width
    ]
    return structured_noise_batch.to(dtype)


def main():
    """Main function to demonstrate structured noise generation."""
    import argparse
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, default="dog.jpg")
    parser.add_argument('--path_out', type=str, default="dog_structured_noise.png")
    parser.add_argument('--cutoff_radii', type=int, nargs='+', default=[10, 20, 40, 80, None])
    parser.add_argument('--sampling_method', type=str, default='fft')
    args = parser.parse_args()
    
    # Load and preprocess image
    image = Image.open(args.path_in)
    image = np.array(image)
    image = image / 255.0
    image = (image - 0.5) * 2.0
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).float().cuda()
    image = image.unsqueeze(0)
    
    # Generate structured noise with different cutoff radii
    cutoff_radii = args.cutoff_radii
    
    for i, cutoff_radius in enumerate(cutoff_radii):
        structured_noise = generate_structured_noise_batch_vectorized(
            image, 
            noise_std=1.0, 
            cutoff_radius=cutoff_radius,
            transition_width=5.0,
            sampling_method=args.sampling_method,
        )
        
        # Post-process for saving
        structured_noise = (structured_noise + 0.5) / 2.0
        structured_noise = structured_noise.permute(0, 2, 3, 1)
        structured_noise = structured_noise.cpu().numpy()[0]
        structured_noise = (structured_noise * 255.0).astype(np.uint8)
        structured_noise = Image.fromarray(structured_noise).convert('RGB')
        
        # Save with cutoff radius in filename
        suffix = f'_cutoff_{cutoff_radius}'
        output_path = args.path_out.replace('.png', f'{suffix}.png')
        structured_noise.save(output_path)
        print(f"Saved structured noise with cutoff radius {cutoff_radius} to {output_path}")
    


if __name__ == "__main__":
    main()