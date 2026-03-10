import cv2
import numpy as np
import matplotlib.pyplot as plt
from aoa import aoa
from image_objective import otsu_multi_objective

def segment_image(image: np.ndarray, thresholds: list) -> np.ndarray:
    """Aplica los umbrales para crear la imagen segmentada."""
    t = np.sort(np.round(thresholds).astype(int))
    segmented = np.zeros_like(image)
    
    # Definir valores de intensidad para cada región (distribuidos uniformemente para visualización)
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
    # Nota: Asegúrate de tener una imagen llamada 'input.jpg' o cambia la ruta
    img_path = 'input.jpg' 
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        # Generar una imagen sintética si no existe el archivo para que el código sea ejecutable
        print(f"No se encontró {img_path}, generando imagen de prueba...")
        image = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(image, (150, 150), 80, 200, -1)
        cv2.rectangle(image, (50, 50), (120, 120), 100, -1)
    
    # 2. Obtener histograma
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    # 3. Configurar AOA para Multi-level Thresholding (ej. 3 umbrales)
    num_thresholds = 3
    bounds = [(0.0, 255.0)] * num_thresholds
    
    print(f"Optimizando {num_thresholds} umbrales con AOA...")
    
    # Ejecutar AOA
    # Pasamos una función lambda que inyecta el histograma en la función objetivo
    best_thresholds, best_val, history = aoa(
        objective_fn=lambda t: otsu_multi_objective(t, hist),
        bounds=bounds,
        n_agents=25,
        max_iter=100,
        verbose=True
    )
    
    best_thresholds = np.sort(np.round(best_thresholds).astype(int))
    print(f"
Umbrales óptimos encontrados: {best_thresholds}")
    print(f"Varianza Otsu máxima: {-best_val:.4f}")

    # 4. Segmentar y Visualizar
    segmented_img = segment_image(image, best_thresholds)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"Segmentada ({num_thresholds} umbrales)")
    plt.imshow(segmented_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Histograma y Umbrales")
    plt.plot(hist)
    for thresh in best_thresholds:
        plt.axvline(x=thresh, color='r', linestyle='--', label=f'T={thresh}')
    plt.xlim([0, 256])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
