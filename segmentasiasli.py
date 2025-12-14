import os
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.filters import roberts, prewitt, sobel
from scipy.ndimage import convolve

# FUNGSI FREI-CHEN
def frei_chen(image):
    sqrt2 = np.sqrt(2)

    k0 = (1/2)*np.array([[1,        sqrt2,  1],
                         [0,        0,      0],
                         [-1,      -sqrt2, -1]])

    k1 = (1/2)*np.array([[1,  0, -1],
                         [sqrt2, 0, -sqrt2],
                         [1,  0, -1]])

    k2 = (1/2)*np.array([[0, 1, 0],
                         [-1, 0, -1],
                         [0, 1, 0]])

    k3 = (1/2)*np.array([[1, -2, 1],
                         [sqrt2, 0, -sqrt2],
                         [1, 2, 1]])

    k4 = (1/2)*np.array([[1, 0, -1],
                         [0, 0, 0],
                         [-1, 0, 1]])

    k5 = (1/2)*np.array([[0, -1, 0],
                         [2, 0, 2],
                         [0, -1, 0]])

    k6 = (1/2)*np.array([[0, 1, 0],
                         [0, 0, 0],
                         [0, -1, 0]])

    k7 = (1/2)*np.array([[1, -2, 1],
                         [0, 0, 0],
                         [-1, 2, -1]])

    k8 = (1/4)*np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])

    kernels = [k0, k1, k2, k3, k4, k5, k6, k7, k8]
    responses = [convolve(image, k) for k in kernels]
    magnitude = np.sqrt(np.sum([r**2 for r in responses], axis=0))
    return magnitude

# Fungsi MSE (self-MSE)
def mse(image):
    image = image.astype(float)
    return np.mean(image ** 2)

# DATASET GAMBAR
image_paths = [
    r"c:/Users/ASUS/Documents/Project Pengcit 2/lanscape 1(grayscale).png",
    r"c:/Users/ASUS/Documents/Project Pengcit 2/potrait 2(grayscale).png",
]

# OUTPUT
output_folder = "hasil_segmentasi"
os.makedirs(output_folder, exist_ok=True)

# PROSES SEMUA GAMBAR
for path in image_paths:

    filename = os.path.basename(path)
    print(f"\n--- Memproses: {filename} ---")

    img = io.imread(path)

    # KONVERSI KE GRAYSCALE
    if img.ndim == 3:
        if img.shape[2] == 4: 
            img = img[:, :, :3]
        img = color.rgb2gray(img)

    # EDGE DETECTION
    edges = {
        "roberts": roberts(img),
        "prewitt": prewitt(img),
        "sobel": sobel(img),
        "frei-chen": frei_chen(img),
    }

    # BUAT FOLDER PENYIMPANAN
    clean_name = filename.replace(".png", "").replace("(", "").replace(")", "").replace(" ", "_")
    save_dir = os.path.join(output_folder, clean_name)
    os.makedirs(save_dir, exist_ok=True)

    # PROSES SETIAP METODE
    for name, edge_img in edges.items():

        # NORMALISASI [0,1]
        mn, mx = np.min(edge_img), np.max(edge_img)
        edge_norm = (edge_img - mn) / (mx - mn + 1e-10)

        # SIMPAN GAMBAR
        edge_uint8 = img_as_ubyte(edge_norm)
        save_path = os.path.join(save_dir, f"{name}.png")
        io.imsave(save_path, edge_uint8)

        # HITUNG MSE
        mse_value = mse(edge_norm)

        print(f"✔ {name:<10} | MSE = {mse_value:.6f} | Disimpan -> {save_path}")

print("\n=== Semua proses selesai ===")
