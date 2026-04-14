import numpy as np
from numba import njit
from typing import Tuple

@njit
def precompute_otsu_data(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Precomputes CDF and Cumulative Weighted Mean for a single channel.
    Ensures strict float64 consistency to avoid Numba TypingErrors.
    """
    # Explicitly cast input to float64
    hist_64 = hist.astype(np.float64)
    
    total_pixels = np.sum(hist_64)
    if total_pixels == 0:
        return np.zeros(256, dtype=np.float64), np.zeros(256, dtype=np.float64), 0.0
    
    # Probabilities (float64)
    p = (hist_64 / total_pixels).flatten()
    
    # CDF (Cumulative Sum of Probabilities)
    cdf = np.cumsum(p).astype(np.float64)
    
    # Cumulative Weighted Mean: sum(i * p[i]) from 0 to k
    intensities = np.arange(256, dtype=np.float64)
    intensity_p = intensities * p
    cum_mean = np.cumsum(intensity_p).astype(np.float64)
    
    # Total mean of the histogram
    mu_total = float(cum_mean[-1])
    
    return cdf, cum_mean, mu_total

@njit
def otsu_multi_objective(thresholds: np.ndarray, cdf: np.ndarray, cum_mean: np.ndarray, mu_total: float) -> float:
    """
    Calculates Otsu's Between-Class Variance for N thresholds in O(N) time.
    Maintains float64 precision for all internal calculations.
    """
    # Thresholds are rounded to nearest integer for histogram indexing
    t = np.unique(np.round(thresholds).astype(np.int32))
    t = np.clip(t, 1, 254)
    
    # Define class borders: [0, t1, t2, ..., tn, 255]
    bins = np.zeros(len(t) + 2, dtype=np.int32)
    bins[1:-1] = t
    bins[-1] = 255
    
    between_class_variance = 0.0 # float64 by default in Numba
    
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i+1]
        
        # Calculate Weight (omega) in O(1)
        if start == 0:
            weight = cdf[end]
            sum_weighted = cum_mean[end]
        else:
            weight = cdf[end] - cdf[start]
            sum_weighted = cum_mean[end] - cum_mean[start]
            
        if weight <= 1e-12: # Increased precision for zero-check
            continue
            
        mean = sum_weighted / weight
        
        # Contribution to between-class variance
        diff = mean - mu_total
        between_class_variance += weight * (diff * diff)
        
    # Minimize the negative variance to maximize original criterion
    return -float(between_class_variance)

@njit
def rgb_otsu_objective(combined_thresholds: np.ndarray, cdfs: np.ndarray, cum_means: np.ndarray, mu_totals: np.ndarray, k: int) -> float:
    """
    Objective function for RGB: Sum of between-class variances across R, G, B channels.
    combined_thresholds: flattened array of size 3*k (k thresholds per channel)
    """
    total_score = 0.0
    for c in range(3):
        # Extract thresholds for the current channel
        t_channel = combined_thresholds[c*k : (c+1)*k]
        # Calculate variance for this channel using precomputed float64 data
        score = otsu_multi_objective(t_channel, cdfs[c], cum_means[c], mu_totals[c])
        total_score += score
        
    return total_score
