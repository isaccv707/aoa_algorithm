import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_segmentation_metrics(original: np.ndarray, segmented: np.ndarray) -> dict:
    """
    Calculates PSNR and SSIM between the original image and its segmented version.
    Supports both Grayscale and RGB images.
    """
    # PSNR
    psnr_val = psnr(original, segmented, data_range=255)
    
    # SSIM
    if original.ndim == 3: # RGB
        # multichannel=True for newer skimage, or channel_axis for version 0.19+
        ssim_val = ssim(original, segmented, data_range=255, channel_axis=-1)
    else: # Grayscale
        ssim_val = ssim(original, segmented, data_range=255)
        
    return {
        "psnr": psnr_val,
        "ssim": ssim_val
    }
