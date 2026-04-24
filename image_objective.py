import numpy as np
from numba import njit
from typing import Tuple

@njit
def precompute_otsu_data(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calcula previamente la función de distribución acumulativa (CDF) y la media ponderada acumulativa para un solo canal.
    Garantiza una estricta consistencia de float64 para evitar errores de tipo de Numba.
    """
    # Convierte la entrada a float64
    hist_64 = hist.astype(np.float64)
    
    total_pixels = np.sum(hist_64)
    if total_pixels == 0:
        return np.zeros(256, dtype=np.float64), np.zeros(256, dtype=np.float64), 0.0
    
    # Probabilidades (float64)
    p = (hist_64 / total_pixels).flatten()
    
    # CDF Suma acumulada de probabilidades
    cdf = np.cumsum(p).astype(np.float64)
    
    # Media ponderada acumulada: sum(i * p[i]) from 0 to k
    intensities = np.arange(256, dtype=np.float64)
    intensity_p = intensities * p
    cum_mean = np.cumsum(intensity_p).astype(np.float64)
    
    # Media total del histograma
    mu_total = float(cum_mean[-1])
    
    return cdf, cum_mean, mu_total

@njit
def otsu_multi_objective(thresholds: np.ndarray, cdf: np.ndarray, cum_mean: np.ndarray, mu_total: float) -> float:
    """
    Calcula la varianza entre clases de Otsu para N umbrales en tiempo O(N).
    Mantiene precisión float64 para todos los cálculos internos.
    """
    t = np.unique(np.round(thresholds).astype(np.int32))
    t = np.clip(t, 1, 254)
    
    bins = np.zeros(len(t) + 2, dtype=np.int32)
    bins[1:-1] = t
    bins[-1] = 255
    
    between_class_variance = 0.0 
    
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i+1]
        
        if start == 0:
            weight = cdf[end]
            sum_weighted = cum_mean[end]
        else:
            weight = cdf[end] - cdf[start]
            sum_weighted = cum_mean[end] - cum_mean[start]
            
        if weight <= 1e-12:
            continue
            
        mean = sum_weighted / weight
        
        diff = mean - mu_total
        between_class_variance += weight * (diff * diff)
        
    return -float(between_class_variance)

@njit
def rgb_otsu_objective(combined_thresholds: np.ndarray, cdfs: np.ndarray, cum_means: np.ndarray, mu_totals: np.ndarray, k: int) -> float:
    """
    Función objetivo para RGB: Suma de las varianzas entre clases en los canales R, G y B.
    Umbrales combinados: matriz aplanada de tamaño 3*k (k umbrales por canal).
    """
    total_score = 0.0
    for c in range(3):
        t_channel = combined_thresholds[c*k : (c+1)*k]
        score = otsu_multi_objective(t_channel, cdfs[c], cum_means[c], mu_totals[c])
        total_score += score
        
    return total_score
