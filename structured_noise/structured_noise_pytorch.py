#!/usr/bin/env python3
"""
PyTorch Structured Noise Generator with Frequency Soft Cutoff

This module provides functions to generate structured noise by combining
Gaussian noise magnitude with image phase using frequency soft cutoff.
"""

import torch
import numpy as np
from PIL import Image


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
    
def save_noise_as_image(noise: torch.Tensor, path: str):
    noise = (noise + 0.5) / 2.0
    noise = noise.permute(0, 2, 3, 1)
    noise = noise.cpu().numpy()[0]
    noise = (noise * 255.0).astype(np.uint8)
    noise = Image.fromarray(noise).convert('RGB')
    noise.save(path)

def main_video():
    import cv2
    import argparse
    import numpy as np
    from PIL import Image
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, default="../town_videos/dog720p_5s.mp4")
    parser.add_argument('--path_out', type=str, default="dog_structured_noise_video.mp4")
    parser.add_argument('--cutoff_radius', type=float, default=40)
    parser.add_argument('--sampling_method', type=str, default='fft')
    args = parser.parse_args()
    video_path = args.path_in
    H, W = 256, 256
    cap = cv2.VideoReader(video_path) if hasattr(cv2, "VideoReader") else cv2.VideoCapture(video_path)
    out_writer = cv2.VideoWriter(
        args.path_out,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (H, W)
    )

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = []
    gray_frames = []

    # ---- Load a few frames ----
    max_frames = 30  # keep it small for testing
    while len(frames) < max_frames:
        ret, frame = cap.read() if isinstance(cap, cv2.VideoCapture) else cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)

    cap.release()
    if len(frames) < 2:
        print("Need at least 2 frames")
        return

    # ---- Resize to something manageable (e.g., 64x64) ----
    gray_resized = [
        cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        for g in gray_frames
    ]

    # ---- Loop over frames ----
    for t in range(len(gray_resized)):
        curr_gray = gray_resized[t]
        # Normalize frame
        curr_gray_normalized = curr_gray / 255.0
        curr_gray_normalized = (curr_gray_normalized - 0.5) * 2.0
        curr_tensor = torch.from_numpy(curr_gray_normalized).unsqueeze(0).unsqueeze(0).to(device)  # (B=1, C=1, H, W)
        
        noise = generate_structured_noise_batch_vectorized(
            curr_tensor,
            noise_std=1.0,
            pad_factor=1.5,
            cutoff_radius=args.cutoff_radius,
            transition_width=2.0,
            sampling_method='fft',
        )

        # Post-process for saving
        noise = (noise + 0.5) / 2.0  # Denormalize
        noise = noise.squeeze(0).squeeze(0)  # (H, W)
        noise_np = noise.cpu().numpy()
        noise_np = (noise_np * 255.0).astype(np.uint8)
        
        # Convert grayscale to BGR for video writer
        noise_bgr = cv2.cvtColor(noise_np, cv2.COLOR_GRAY2BGR)
        
        out_writer.write(noise_bgr)

    out_writer.release()
    print(f"Saved structured noise video to {args.path_out}")


def main_video_warp():
    import cv2
    import argparse
    import numpy as np
    from PIL import Image
    import os
    import sys
    import taichi as ti

    # Add the path to the warping code to the system path
    from warp_particle import ParticleWarper

    ti.init(arch=ti.gpu, device_memory_GB=4.0, debug=False, default_fp=ti.f64, random_seed=0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, default="../town_videos/dog720p_5s.mp4")
    parser.add_argument('--path_out', type=str, default="dog_structured_noise_video_warped.mp4")
    parser.add_argument('--cutoff_radius', type=float, default=20)
    parser.add_argument('--sampling_method', type=str, default='fft')
    args = parser.parse_args()
    video_path = args.path_in
    H, W = 256, 256
    cap = cv2.VideoCapture(video_path)
    out_writer = cv2.VideoWriter(
        args.path_out,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (W, H)
    )

    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = []
    gray_frames = []

    # ---- Load a few frames ----
    max_frames = 90  # keep it small for testing
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)

    cap.release()
    if len(frames) < 2:
        print("Need at least 2 frames")
        return

    # ---- Generate initial noise from the first frame ----
    first_frame_normalized = (frames[0].astype(np.float32) / 255.0 - 0.5) * 2.0
    first_frame_tensor = torch.from_numpy(first_frame_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    # Using 3 channels for the initial noise for color video
    initial_noise = generate_structured_noise_batch_vectorized(
        first_frame_tensor,
        noise_std=1.0,
        pad_factor=1.5,
        cutoff_radius=args.cutoff_radius,
        transition_width=2.0,
        sampling_method=args.sampling_method,
    )
    
    # The warper works with numpy arrays on CPU
    prev_noise = initial_noise.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Save first frame of noise
    noise_to_save = (prev_noise - prev_noise.min()) / (prev_noise.max() - prev_noise.min())
    noise_to_save = (noise_to_save * 255).astype(np.uint8)
    out_writer.write(cv2.cvtColor(noise_to_save, cv2.COLOR_RGB2BGR))

    # ---- Setup for warping ----
    n_noise_channels = prev_noise.shape[2]
    warper = ParticleWarper(H, W, n_noise_channels)

    # identity maps (cell center)
    i = np.arange(H) + 0.5
    j = np.arange(W) + 0.5
    ii, jj = np.meshgrid(i, j, indexing='ij')
    identity_cc = np.stack((jj, ii), axis=-1) # Note: meshgrid and opencv use (x,y) which is (W,H)

    # ---- Loop over frames and warp noise ----
    for t in range(1, len(gray_frames)):
        print(f"Processing frame {t}")
        prev_gray = gray_frames[t-1]
        curr_gray = gray_frames[t]

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # The warper expects a map of where each grid point *comes from*.
        # Flow is (dx, dy), so new_pos = old_pos + flow.
        # The deformation map should be old_pos = new_pos - flow.
        # Here, new_pos is `identity_cc`.
        # The flow from Farneback is how pixels move from prev to curr.
        # So a pixel at (x,y) in prev is at (x+dx, y+dy) in curr.
        # The deformation map for the warper should be `identity_cc - flow`.
        # The `map_field` in the kernel is where the destination pixel `(i,j)` comes from in the source.
        # So we need backward flow. `calcOpticalFlowFarneback` computes forward flow.
        # For warping, we need to know for each pixel in the target image, where it comes from in the source.
        # Let's use the forward flow as an approximation of the backward flow for simplicity,
        # which is not entirely correct but often works for small motions.
        # A better way would be to compute backward flow, or use the forward flow to construct a backward map.
        # Let's stick to the logic from the example: `identity_cc - flow_map_i`
        
        # The flow from opencv is (dx, dy) for each pixel.
        # The shape is (H, W, 2).
        # `identity_cc` is (H, W, 2) with (x, y) coordinates.
        # The deformation map should be where each pixel of the new image comes from in the old one.
        # `map_field[i,j]` is the coordinate in the old image that maps to `(i,j)` in the new image.
        # `flow[y,x]` gives the displacement `(dx, dy)`. A pixel at `(x,y)` in `prev_gray` moves to `(x+dx, y+dy)` in `curr_gray`.
        # So, a pixel at `(x', y')` in `curr_gray` comes from `(x'-dx, y'-dy)` in `prev_gray`.
        # The flow we have is `flow(x,y)`. We need `flow(x', y')`. We approximate `flow(x',y')` with `flow(x,y)`.
        # So the source coordinate is `(x,y) - flow(x,y)`.
        # The `identity_cc` has shape (H, W, 2) and stores `(x,y)` coordinates.
        # The `flow` has shape (H, W, 2) and stores `(dx,dy)`.
        # The deformation map is `identity_cc - flow`.
        # Note on indexing: `meshgrid` with `indexing='ij'` gives `ii` with shape (H,W) and `jj` with shape (H,W).
        # `ii` has row indices, `jj` has column indices.
        # OpenCV flow `(dx, dy)` corresponds to `(d_col, d_row)`.
        # `identity_cc` was created with `(jj, ii)` so it's `(x,y)` i.e. `(col, row)`.
        # So `identity_cc - flow` should be correct.
        
        # The `warp_particle` code uses `(i,j)` as `(row, col)`.
        # Let's re-create identity_cc to be sure.
        i_coords = np.arange(H) + 0.5 # rows
        j_coords = np.arange(W) + 0.5 # cols
        jj, ii = np.meshgrid(j_coords, i_coords) # jj is x, ii is y
        identity_map = np.stack((ii, jj), axis=-1)

        # flow from opencv is also (dx, dy) where dx is change in x (columns), dy is change in y (rows)
        # But the order in the last dim is (dx, dy).
        # The `map_field` in `warp_particle` is indexed by `(i,j)` which is `(row, col)`.
        # And it expects `(y,x)` coordinates.
        # `warped_pos = map_field[i, j]-0.5`
        # `lower_corner = ti.math.floor(warped_pos)`
        # `lower_x, lower_y = int(lower_corner.x), int(lower_corner.y)`
        # Here `x` is the first component, `y` is the second.
        # So `map_field` should store `(row_coord, col_coord)`.
        # `identity_map` created above has `(y,x)` i.e. `(row, col)` coords.
        # The flow from opencv has `(dx, dy)`. We need to swap it to `(dy, dx)` to match our `identity_map`.
        flow_swapped = np.stack((flow[..., 1], flow[..., 0]), axis=-1)
        
        deformation_map = identity_map - flow_swapped

        warper.set_deformation(deformation_map)
        warper.set_noise(prev_noise)
        warper.run()
        warped_noise = warper.noise_field.to_numpy()

        # Post-process for saving
        noise_to_save = (warped_noise - warped_noise.min()) / (warped_noise.max() - warped_noise.min())
        noise_to_save = (noise_to_save * 255).astype(np.uint8)
        out_writer.write(cv2.cvtColor(noise_to_save, cv2.COLOR_RGB2BGR))

        prev_noise = warped_noise

    out_writer.release()
    print(f"Saved structured noise video to {args.path_out}")


if __name__ == "__main__":
    # main_video()
    main_video_warp()