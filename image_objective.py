import numpy as np
from typing import List

def otsu_multi_objective(thresholds: List[float], hist: np.ndarray) -> float:
    """
    Calcula la Varianza Entre-Clases de Otsu para N umbrales.
    Minimiza el negativo de la varianza para que el AOA maximice el criterio original.
    """
    # 1. Adaptación y limpieza de umbrales
    # Redondeamos a enteros y aseguramos que estén en orden ascendente y únicos
    t = np.unique(np.sort(np.round(thresholds).astype(int)))
    
    # Asegurar que los umbrales estén dentro de [1, 254] para evitar clases fuera de rango
    t = np.clip(t, 1, 254)
    
    total_pixels = hist.sum()
    if total_pixels == 0: return 0.0
    p = (hist / total_pixels).flatten() # Normalizar histograma
    
    # Definir los bordes de los niveles: [0, t1, t2, ..., tn, 256]
    bins = np.concatenate(([0], t, [256]))
    
    mu_total = np.sum(np.arange(256) * p)
    between_class_variance = 0.0
    
    for i in range(len(bins) - 1):
        start, end = int(bins[i]), int(bins[i+1])
        if start >= end: continue
        
        weight = np.sum(p[start:end])
        if weight <= 0: continue
        
        mean = np.sum(np.arange(start, end) * p[start:end]) / weight
        between_class_variance += weight * (mean - mu_total) ** 2
        
    return -float(between_class_variance)
