import os
import imageio
import torch
import numpy as np
from typing import Dict, Any
from torch import Tensor

def compress_png(compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, bit: int = 8) -> Dict[str, Any]:
    """
    Compress parameters using PNG with specified bit quantization.
    
    Args:
        compress_dir (str): Directory to save the compressed image(s).
        param_name (str): Base filename for the parameter images.
        params (Tensor): The tensor to compress.
        n_sidelen (int): The side length for reshaping the tensor.
        bit (int): Bit depth for quantization (8, 16, or 32).
        
    Returns:
        Dict[str, Any]: Metadata including original shape, data type, and normalization parameters.
    """
    # If tensor is empty, just store meta
    if params.numel() == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
            "bit": bit
        }
        return meta

    # Reshape tensor into a grid and compute min/max for normalization
    grid = params.reshape((n_sidelen, n_sidelen, -1)).squeeze()
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)

    # Convert normalized grid to numpy array
    img_norm = grid_norm.detach().cpu().numpy()

    max_val = 2**bit - 1
    if bit == 8:
        # Quantize and save a single 8-bit image
        img = (img_norm * max_val).round().astype(np.uint8)
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)
    elif bit in (16, 32):
        # Quantize the image according to the bit depth
        if bit == 16:
            dtype = np.uint16
            num_chunks = 2
        else:  # bit == 32
            dtype = np.uint32
            num_chunks = 4

        img = (img_norm * max_val).round().astype(dtype)

        # Save each 8-bit chunk using unified naming: param_name_0.png, param_name_1.png, ...
        chunks = []
        for i in range(num_chunks):
            # Extract 8 bits at a time; i=0 gives the lowest order 8 bits
            chunk = ((img >> (8 * i)) & 0xFF).astype(np.uint8)
            chunks.append(chunk)
            imageio.imwrite(os.path.join(compress_dir, f"{param_name}_{i}.png"), chunk)
    else:
        raise ValueError("Unsupported bit depth. Please use 8, 16, or 32.")
    
    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "bit": bit,
    }
    return meta

def decompress_png(compress_dir: str, param_name: str, meta: dict) -> np.ndarray:
    """
    Decompress parameters from PNG(s) with specified bit quantization.
    
    Args:
        compress_dir (str): Directory where the compressed image(s) are stored.
        param_name (str): Base filename for the parameter images.
        meta (dict): Metadata produced during compression.
        
    Returns:
        np.ndarray: The decompressed parameters with original shape and data type.
    """
    bit = meta["bit"]
    max_val = 2**bit - 1
    if bit == 8:
        # Read single 8-bit image
        img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png")).astype(np.float32)
    elif bit in (16, 32):
        if bit == 16:
            num_chunks = 2
        else:  # bit == 32
            num_chunks = 4

        # Initialize the image with zeros; determine shape from one of the chunks
        first_chunk = imageio.imread(os.path.join(compress_dir, f"{param_name}_0.png")).astype(np.uint32)
        img = np.zeros_like(first_chunk, dtype=np.uint32)
        
        # Combine each 8-bit chunk into a single integer value
        for i in range(num_chunks):
            chunk = imageio.imread(os.path.join(compress_dir, f"{param_name}_{i}.png")).astype(np.uint32)
            img |= (chunk << (8 * i))
        img = img.astype(np.float32)
    else:
        raise ValueError("Unsupported bit depth. Please use 8, 16, or 32.")
    
    # Normalize back to [0, 1]
    img_norm = img / max_val

    # Denormalize using the stored minimum and maximum values
    mins = np.array(meta["mins"], dtype=np.float32)
    maxs = np.array(meta["maxs"], dtype=np.float32)
    grid = img_norm * (maxs - mins) + mins

    # Reshape to original parameters shape and cast to the stored dtype
    params = np.reshape(grid, meta["shape"])
    params = params.astype(meta["dtype"])
    return params
