import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from aoa import aoa
from image_objective import otsu_multi_objective

def segment_image(image: np.ndarray, thresholds: list) -> np.ndarray:
    """Aplica los umbrales para crear la imagen segmentada."""
    t = np.sort(np.round(thresholds).astype(int))
    segmented = np.zeros_like(image)
    
    # Definir valores de intensidad para cada región (distribuidos uniformemente)
    vals = np.linspace(0, 255, len(t) + 1, dtype=np.uint8)
    
    # Caso 1: Debajo del primer umbral
    segmented[image < t[0]] = vals[0]
    
    # Casos intermedios
    for i in range(len(t) - 1):
        mask = (image >= t[i]) & (image < t[i+1])
        segmented[mask] = vals[i+1]
        
    # Caso último: Por encima del último umbral
    segmented[image >= t[-1]] = vals[-1]
    
    return segmented

def main():
    # 1. Cargar imagen en escala de grises
    img_path = 'input.jpg' 
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"No se encontró {img_path}, generando imagen de prueba...")
        image = np.zeros((400, 400), dtype=np.uint8)
        # Crear un gradiente y algunas formas para segmentar
        cv2.rectangle(image, (0, 0), (400, 400), 50, -1)
        cv2.circle(image, (200, 200), 120, 150, -1)
        cv2.circle(image, (200, 200), 60, 250, -1)
        cv2.rectangle(image, (50, 50), (120, 120), 100, -1)
        # Añadir un poco de ruido
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Pre-procesamiento: Histograma para AOA
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # --- MÉTODO 1: K-Means (4 clústeres) ---
    print("\nEjecutando K-Means...")
    start_time = time.time()
    
    pixel_values = image.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    kmeans_segmented = centers[labels.flatten()].reshape(image.shape)
    
    kmeans_time = time.time() - start_time

    # --- MÉTODO 2: Otsu Clásico (OpenCV - 1 umbral) ---
    print("Ejecutando Otsu OpenCV...")
    start_time = time.time()
    
    otsu_thresh, otsu_segmented = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    otsu_time = time.time() - start_time

    # --- MÉTODO 3: AOA Multinivel (3 umbrales -> 4 segmentos) ---
    print("Ejecutando AOA Multinivel...")
    num_thresholds = 3
    bounds = [(0.0, 255.0)] * num_thresholds
    
    start_time = time.time()
    
    best_thresholds, _, _ = aoa(
        objective_fn=lambda t: otsu_multi_objective(t, hist),
        bounds=bounds,
        n_agents=25,
        max_iter=100,
        verbose=False
    )
    
    aoa_segmented = segment_image(image, best_thresholds)
    aoa_time = time.time() - start_time

    # --- CÁLCULO DE MÉTRICAS (PSNR y SSIM) ---
    print("\nCalculando métricas de calidad...")
    psnr_k = calc_psnr(image, kmeans_segmented, data_range=255)
    ssim_k = calc_ssim(image, kmeans_segmented, data_range=255)
    
    psnr_o = calc_psnr(image, otsu_segmented, data_range=255)
    ssim_o = calc_ssim(image, otsu_segmented, data_range=255)
    
    psnr_a = calc_psnr(image, aoa_segmented, data_range=255)
    ssim_a = calc_ssim(image, aoa_segmented, data_range=255)

    # --- IMPRIMIR TABLA EN CONSOLA ---
    print("\n" + "="*65)
    print(f"{'MÉTODO':<18} | {'TIEMPO (s)':<12} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 65)
    print(f"{'K-Means (k=4)':<18} | {kmeans_time:<12.4f} | {psnr_k:<12.2f} | {ssim_k:<10.4f}")
    print(f"{'Otsu Clásico':<18} | {otsu_time:<12.4f} | {psnr_o:<12.2f} | {ssim_o:<10.4f}")
    print(f"{'AOA Multinivel':<18} | {aoa_time:<12.4f} | {psnr_a:<12.2f} | {ssim_a:<10.4f}")
    print("="*65)

    # --- VISUALIZACIÓN Y GUARDADO ---
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title("Imagen Original (Grises)")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f"K-Means (k=4)\nTiempo: {kmeans_time:.3f}s | PSNR: {psnr_k:.1f} | SSIM: {ssim_k:.2f}")
    plt.imshow(kmeans_segmented, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f"Otsu OpenCV (1 umbral)\nTiempo: {otsu_time:.3f}s | PSNR: {psnr_o:.1f} | SSIM: {ssim_o:.2f}")
    plt.imshow(otsu_segmented, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f"AOA Multinivel (3 umbrales)\nTiempo: {aoa_time:.3f}s | PSNR: {psnr_a:.1f} | SSIM: {ssim_a:.2f}")
    plt.imshow(aoa_segmented, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    
    # --- Lógica infalible para guardar en la misma carpeta ---
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(carpeta_actual, "resultado_comparativa.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*65)
    print("¡LA IMAGEN SE GUARDÓ EXACTAMENTE AQUÍ:")
    print(output_filename)
    print("="*65)

if __name__ == "__main__":
    main()