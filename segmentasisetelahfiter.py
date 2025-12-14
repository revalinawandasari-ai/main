# IMPORT LIBRARIES
import os
import numpy as np
from skimage import io, img_as_float, img_as_ubyte, color
from scipy.ndimage import minimum_filter, maximum_filter, median_filter, uniform_filter, convolve
from skimage.filters import roberts, prewitt, sobel

# FUNGSI FREI-CHEN
def frei_chen(image):
    k0 = (1/2)*np.array([[1, np.sqrt(2), 1],
                         [0, 0, 0],
                         [-1, -np.sqrt(2), -1]])

    k1 = (1/2)*np.array([[1, 0, -1],
                         [np.sqrt(2), 0, -np.sqrt(2)],
                         [1, 0, -1]])

    k2 = (1/2)*np.array([[0, 1, 0],
                         [-1, 0, -1],
                         [0, 1, 0]])

    k3 = (1/2)*np.array([[1, -2, 1],
                         [0, 0, 0],
                         [-1, 2, -1]])

    k4 = (1/2)*np.array([[0, -1, 0],
                         [2, 0, 2],
                         [0, -1, 0]])

    k5 = (1/2)*np.array([[1, 0, -1],
                         [0, 0, 0],
                         [-1, 0, 1]])

    k6 = (1/2)*np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, -1, 0]])

    k7 = (1/2)*np.array([[1, -2, 1],
                         [0, 0, 0],
                         [-1, 2, -1]])

    kernels = [k0, k1, k2, k3, k4, k5, k6, k7]
    responses = [convolve(image, k) for k in kernels]

    magnitude = np.sqrt(np.sum([r**2 for r in responses], axis=0))
    return magnitude

# PATH GAMBAR
image_paths = [
    "c:/Users/ASUS/Documents/Project Pengcit 2/Salt & Pepper Level 1 (0.02) - Grayscale 1.png",
    "c:/Users/ASUS/Documents/Project Pengcit 2/Salt & Pepper Level 1 (0.02) - Grayscale 2.png",
    "c:/Users/ASUS/Documents/Project Pengcit 2/Gaussian Level 1 (0, 10) - Grayscale 1.png",
    "c:/Users/ASUS/Documents/Project Pengcit 2/Gaussian Level 1 (0, 10) - Grayscale 2.png"
]

# FUNGSI FILTER
def apply_filters(image):
    filters_dict = {
        "min_filter": minimum_filter(image, size=3),
        "max_filter": maximum_filter(image, size=3),
        "mean_filter": uniform_filter(image, size=3),
        "median_filter": median_filter(image, size=3)
    }
    return filters_dict

# FUNGSI EDGE DETECTION
def apply_edge_detection(image):
    edges = {
        "roberts": roberts(image),
        "prewitt": prewitt(image),
        "sobel": sobel(image),
        "frei_chen": frei_chen(image)
    }
    return edges

# FUNGSI MSE
def mse(img):
    img = img_as_float(img)
    return np.mean(img ** 2)

# MAIN PROGRAM
for path in image_paths:
    file_name = os.path.basename(path)
    print(f"\n--- Memproses: {file_name} ---")

    # LOAD IMAGE (Grayscale / RGB / RGBA)
    raw = io.imread(path)

    if raw.ndim == 2:
        img = img_as_float(raw)

    elif raw.ndim == 3 and raw.shape[-1] == 3:
        img = img_as_float(color.rgb2gray(raw))

    elif raw.ndim == 3 and raw.shape[-1] == 4:
        raw = raw[:, :, :3]
        img = img_as_float(color.rgb2gray(raw))

    else:
        raise ValueError("Format gambar tidak dikenali.")

    # 1. FILTERING
    filtered_images = apply_filters(img)

    # OUTPUT FOLDER PER GAMBAR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "hasil_output", file_name.replace(".png", ""))
    os.makedirs(output_dir, exist_ok=True)

    # 2. EDGE DETECTION + MSE
    for filter_name, filtered_img in filtered_images.items():

        edges = apply_edge_detection(filtered_img)

        for method_name, edge_img in edges.items():

            # NORMALISASI
            edge_norm = (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min())

            # convert ke uint8
            edge_uint8 = img_as_ubyte(edge_norm)

            # simpan file
            save_path = os.path.join(output_dir, f"{filter_name}_{method_name}.png")
            io.imsave(save_path, edge_uint8)

            # HITUNG MSE GAMBAR HASIL
            mse_value = mse(edge_uint8)

            print(f"✔ Disimpan -> {save_path}")
            print(f"   MSE = {mse_value}\n")