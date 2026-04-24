import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_segmentation_metrics(original: np.ndarray, segmented: np.ndarray) -> dict:
    """
    Calcula la relación señal-ruido (PSNR) y el índice de intensidad de señal (SSIM) entre la imagen original y su versión segmentada.
    Admite imágenes en escala de grises y RGB.
    """
    # PSNR
    psnr_val = psnr(original, segmented, data_range=255)

    # SSIM
    if original.ndim == 3: # RGB
        ssim_val = ssim(original, segmented, data_range=255, channel_axis=-1)
    else: # Escala de grises
        ssim_val = ssim(original, segmented, data_range=255)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val
    }
