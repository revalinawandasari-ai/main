# Import library
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, color, filters, exposure, util
from scipy import ndimage as ndi

# Menentukan PATH Citra
image_paths = [
    "c:/Users/ASUS/Documents/Project Pengcit 2/Salt & Pepper Level 1 (0.02) - Grayscale 1.png",
    "c:/Users/ASUS/Documents/Project Pengcit 2/Gaussian Level 1 (0, 10) - Grayscale 1.png",
    "c:/Users/ASUS/Documents/Project Pengcit 2/Salt & Pepper Level 1 (0.02) - Grayscale 2.png",
    "c:/Users/ASUS/Documents/Project Pengcit 2/Gaussian Level 1 (0, 10) - Grayscale 2.png",
]

# Membuat folder output
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

# Fungsi load grayscale
def load_gray(path):
    im = io.imread(path)

    # Jika gambar memiliki 4 channel (RGBA), buang channel alpha
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]   # ambil hanya RGB

    # Jika masih 3 channel (RGB), konversi ke grayscale
    if im.ndim == 3:
        im = color.rgb2gray(im)

    return img_as_float(im)

# Definisi metode segmentasi (edge detection)
def roberts_edge(im):
    return filters.roberts(im)

def prewitt_edge(im):
    return filters.prewitt(im)

def sobel_edge(im):
    return filters.sobel(im)

def frei_chen_edge(im):
    sqrt2 = np.sqrt(2.0)
    m1 = np.array([[ 1,  sqrt2,  1],
                   [ 0,      0,  0],
                   [-1, -sqrt2, -1]], float)

    m2 = np.array([[ 1,  0, -1],
                   [ sqrt2, 0, -sqrt2],
                   [ 1,  0, -1]], float)

    r1 = ndi.convolve(im, m1, mode="reflect")
    r2 = ndi.convolve(im, m2, mode="reflect")
    mag = np.sqrt(r1**2 + r2**2)
    mag = mag / (mag.max() + 1e-12)
    return mag

detectors = {
    "Roberts": roberts_edge,
    "Prewitt": prewitt_edge,
    "Sobel": sobel_edge,
    "Frei-Chen": frei_chen_edge
}

# Memproses setiap gambar
def process_image(im_path):
    im = load_gray(im_path)
    name = os.path.splitext(os.path.basename(im_path))[0]

    # Menjalankan detektor tepi & menyimpan output
    for method, func in detectors.items():
        res = func(im)
        res = exposure.rescale_intensity(res, out_range=(0, 1))

        plt.figure(figsize=(6, 4))
        plt.imshow(res, cmap="gray")
        plt.title(f"{method} - {name}")
        plt.axis("off")

        save_path = f"{output_dir}/{name}_{method.replace(' ', '_')}.png"
        io.imsave(save_path, util.img_as_ubyte(res))

    # Membuat gambar perbandingan
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(im, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    i = 2
    for method, func in detectors.items():
        res = func(im)
        res = exposure.rescale_intensity(res, out_range=(0, 1))
        plt.subplot(1, 5, i)
        plt.imshow(res, cmap="gray")
        plt.title(method)
        plt.axis("off")
        i += 1

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{name}_comparison.png", dpi=200)
    plt.close()

# Loop semua citra input
for path in image_paths:
    if os.path.exists(path):
        process_image(path)
    else:
        print("File tidak ditemukan:", path)

# Output akhir
print("Selesai. Hasil disimpan di folder:", output_dir)