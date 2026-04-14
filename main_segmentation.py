import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from aoa import aoa
from image_objective import rgb_otsu_objective, precompute_otsu_data
from segmentation_metrics import calculate_segmentation_metrics

def segment_channel(image_channel: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Aplica los umbrales para crear la imagen segmentada de un solo canal."""
    t = np.sort(np.round(thresholds).astype(int))
    segmented = np.zeros_like(image_channel)
    vals = np.linspace(0, 255, len(t) + 1, dtype=np.uint8)
    
    segmented[image_channel < t[0]] = vals[0]
    for i in range(len(t) - 1):
        mask = (image_channel >= t[i]) & (image_channel < t[i+1])
        segmented[mask] = vals[i+1]
    segmented[image_channel >= t[-1]] = vals[-1]
    
    return segmented

def main():
    # 1. Cargar imagen (RGB por defecto)
    img_path = 'input.jpg' 
    image_bgr = cv2.imread(img_path)
    
    if image_bgr is None:
        print(f"No se encontró {img_path}, generando imagen RGB de prueba...")
        image_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(image_bgr, (150, 150), 80, (200, 50, 50), -1) # Blueish
        cv2.rectangle(image_bgr, (50, 50), (120, 120), (50, 200, 50), -1) # Greenish
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Pre-calcular datos para los 3 canales
    cdfs = np.zeros((3, 256))
    cum_means = np.zeros((3, 256))
    mu_totals = np.zeros(3)
    
    for c in range(3):
        hist = cv2.calcHist([image_rgb], [c], None, [256], [0, 256])
        cdfs[c], cum_means[c], mu_totals[c] = precompute_otsu_data(hist)
    
    # 3. Configurar AOA para RGB
    # k umbrales por canal = 3*k dimensiones en total
    k = 3 
    total_dims = 3 * k
    bounds = [(0.0, 255.0)] * total_dims
    
    print(f"Optimizando RGB ({k} umbrales/canal, total {total_dims} dims) con AOA...")
    
    start_time = time.time()
    
    # Ejecutar AOA con el objetivo combinado
    best_thresholds, best_val, history = aoa(
        objective_fn=lambda t: rgb_otsu_objective(t, cdfs, cum_means, mu_totals, k),
        bounds=bounds,
        n_agents=35,
        max_iter=100,
        verbose=False # Silenciamos para el reporte final
    )
    
    execution_time = time.time() - start_time
    
    # 4. Segmentar y Reconstruir
    segmented_rgb = np.zeros_like(image_rgb)
    thresholds_report = []
    
    for c in range(3):
        t_channel = best_thresholds[c*k : (c+1)*k]
        t_sorted = np.sort(np.round(t_channel).astype(int))
        thresholds_report.append(t_sorted)
        segmented_rgb[:, :, c] = segment_channel(image_rgb[:, :, c], t_sorted)
    
    # 5. Calcular Métricas de Calidad
    metrics = calculate_segmentation_metrics(image_rgb, segmented_rgb)
    
    # 6. Reporte Consolidado
    print("\n" + "="*40)
    print("CONSOLIDATED SEGMENTATION REPORT")
    print("="*40)
    print(f"Execution Time:      {execution_time:.4f} seconds")
    print(f"Final Fitness Score: {-best_val:.4f}")
    print(f"PSNR:                {metrics['psnr']:.2f} dB")
    print(f"SSIM:                {metrics['ssim']:.4f}")
    print("-" * 40)
    print("Optimal Thresholds per Channel:")
    print(f"  Red:   {thresholds_report[0]}")
    print(f"  Green: {thresholds_report[1]}")
    print(f"  Blue:  {thresholds_report[2]}")
    print("="*40)

    # 7. Visualización
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original RGB Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Segmented RGB ({k} thresholds/ch)")
    plt.imshow(segmented_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
