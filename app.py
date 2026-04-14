import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from aoa.algorithm import aoa
from image_objective import rgb_otsu_objective, precompute_otsu_data
from segmentation_metrics import calculate_segmentation_metrics

# --- Page Config ---
st.set_page_config(page_title="AOA Image Segmentation", layout="wide")

st.title("AOA Image Segmentation Dashboard")
st.markdown("""
This dashboard uses the **Arithmetic Optimization Algorithm (AOA)** to perform multi-level RGB image segmentation. 
It finds the optimal thresholds for each channel (Red, Green, Blue) by maximizing **Otsu's Between-Class Variance**.
""")

# --- Sidebar: Hyperparameters ---
st.sidebar.header("AOA Hyperparameters")
n_agents = st.sidebar.slider("Number of Agents", 10, 100, 30, help="Population size of the algorithm.")
max_iter = st.sidebar.slider("Maximum Iterations", 10, 500, 100, help="Maximum number of optimization steps.")
k_thresholds = st.sidebar.slider("Thresholds per Channel (K)", 1, 5, 3, help="Number of levels per RGB channel.")

with st.sidebar.expander("About the Metrics"):
    st.markdown("""
    - **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction quality (Higher is better).
    - **SSIM (Structural Similarity Index):** Measures perceived similarity (1.0 is perfect).
    """)

# --- Helper Functions ---
def segment_channel(image_channel, thresholds):
    t = np.sort(np.round(thresholds).astype(int))
    segmented = np.zeros_like(image_channel)
    vals = np.linspace(0, 255, len(t) + 1, dtype=np.uint8)
    
    segmented[image_channel < t[0]] = vals[0]
    for i in range(len(t) - 1):
        mask = (image_channel >= t[i]) & (image_channel < t[i+1])
        segmented[mask] = vals[i+1]
    segmented[image_channel >= t[-1]] = vals[-1]
    return segmented

@st.cache_data
def run_aoa_pipeline(image_rgb, n_agents, max_iter, k):
    # Precompute data for the 3 channels
    cdfs = np.zeros((3, 256))
    cum_means = np.zeros((3, 256))
    mu_totals = np.zeros(3)
    hists = []

    for c in range(3):
        hist = cv2.calcHist([image_rgb], [c], None, [256], [0, 256])
        hists.append(hist)
        cdfs[c], cum_means[c], mu_totals[c] = precompute_otsu_data(hist)
    
    total_dims = 3 * k
    bounds = [(0.0, 255.0)] * total_dims
    
    start_time = time.time()
    best_thresholds, best_val, history = aoa(
        objective_fn=lambda t: rgb_otsu_objective(t, cdfs, cum_means, mu_totals, k),
        bounds=bounds,
        n_agents=n_agents,
        max_iter=max_iter,
        verbose=False
    )
    execution_time = time.time() - start_time
    
    # Reconstruct Segmented Image
    segmented_rgb = np.zeros_like(image_rgb)
    thresholds_report = []
    for c in range(3):
        t_channel = best_thresholds[c*k : (c+1)*k]
        t_sorted = np.sort(np.round(t_channel).astype(int))
        thresholds_report.append(t_sorted)
        segmented_rgb[:, :, c] = segment_channel(image_rgb[:, :, c], t_sorted)
        
    metrics = calculate_segmentation_metrics(image_rgb, segmented_rgb)
    
    return {
        "segmented_rgb": segmented_rgb,
        "best_val": -best_val,
        "history": history,
        "execution_time": execution_time,
        "thresholds": thresholds_report,
        "metrics": metrics,
        "hists": hists
    }

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)

    if st.button("Run AOA Optimization"):
        with st.spinner("Optimizing thresholds... This may take a few seconds."):
            results = run_aoa_pipeline(image_rgb, n_agents, max_iter, k_thresholds)
        
        with col2:
            st.subheader("Segmented Result")
            st.image(results["segmented_rgb"], use_container_width=True)

        # Metrics display
        st.write("---")
        st.subheader("Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution Time", f"{results['execution_time']:.3f}s")
        m2.metric("Best Fitness", f"{results['best_val']:.2f}")
        m3.metric("PSNR", f"{results['metrics']['psnr']:.2f} dB")
        m4.metric("SSIM", f"{results['metrics']['ssim']:.4f}")

        # Analysis Layout
        st.write("---")
        tab1, tab2 = st.tabs(["Convergence Curve", "RGB Histograms"])

        with tab1:
            st.subheader("AOA Optimization Progress")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            ax_hist.plot(results["history"], color='purple', linewidth=2)
            ax_hist.set_xlabel("Iteration")
            ax_hist.set_ylabel("Fitness (Otsu Variance)")
            ax_hist.grid(True, alpha=0.3)
            st.pyplot(fig_hist)

        with tab2:
            st.subheader("Optimal Thresholds per Channel")
            colors = ['red', 'green', 'blue']
            fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4))
            for i, color in enumerate(colors):
                axes_h[i].plot(results["hists"][i], color=color)
                axes_h[i].set_title(f"{color.capitalize()} Channel")
                for t in results["thresholds"][i]:
                    axes_h[i].axvline(x=t, color='black', linestyle='--', alpha=0.7)
                axes_h[i].set_xlim([0, 256])
            st.pyplot(fig_h)
            
            st.info(f"**Thresholds Found:** R: {results['thresholds'][0]} | G: {results['thresholds'][1]} | B: {results['thresholds'][2]}")

else:
    st.info("Please upload an image to start the segmentation process.")

st.sidebar.markdown("---")
st.sidebar.write("Developed with Python, Numba & Streamlit.")
