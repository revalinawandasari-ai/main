import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi Salt & Pepper Noise (manual)
def salt_pepper_noise(img, prob):
    noisy = img.copy()
    h, w = img.shape[:2]
    rand = np.random.rand(h, w)

    # Mask untuk salt / pepper
    salt_mask = rand < (prob / 2)
    pepper_mask = rand > 1 - (prob / 2)

    if img.ndim == 2:  # grayscale
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
    else:  # RGB
        noisy[salt_mask, :] = 255
        noisy[pepper_mask, :] = 0

    return noisy

# Fungsi Gaussian Noise (manual)
def gaussian_noise(img, mean, sigma):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# Filter helpers: semua mengembalikan uint8
def _pad_image(img, pad):
    if img.ndim == 2:
        return np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    else:
        return np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

def min_filter(img, k=5):
    h, w = img.shape[:2]
    pad = k // 2
    padded = _pad_image(img, pad)
    output = np.zeros_like(img, dtype=np.float64)

    if img.ndim == 2:
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k, j:j+k]
                output[i, j] = np.min(window)
    else:
        for i in range(h):
            for j in range(w):
                for c in range(3):
                    window = padded[i:i+k, j:j+k, c]
                    output[i, j, c] = np.min(window)

    return np.clip(output, 0, 255).astype(np.uint8)

def max_filter(img, k=5):
    h, w = img.shape[:2]
    pad = k // 2
    padded = _pad_image(img, pad)
    output = np.zeros_like(img, dtype=np.float64)

    if img.ndim == 2:
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k, j:j+k]
                output[i, j] = np.max(window)
    else:
        for i in range(h):
            for j in range(w):
                for c in range(3):
                    window = padded[i:i+k, j:j+k, c]
                    output[i, j, c] = np.max(window)

    return np.clip(output, 0, 255).astype(np.uint8)

def median_filter(img, k=5):
    h, w = img.shape[:2]
    pad = k // 2
    padded = _pad_image(img, pad)
    output = np.zeros_like(img, dtype=np.float64)

    if img.ndim == 2:
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k, j:j+k].ravel()
                output[i, j] = np.median(window)
    else:
        for i in range(h):
            for j in range(w):
                for c in range(3):
                    window = padded[i:i+k, j:j+k, c].ravel()
                    output[i, j, c] = np.median(window)

    return np.clip(output, 0, 255).astype(np.uint8)

def mean_filter(img, k=5):
    h, w = img.shape[:2]
    pad = k // 2
    padded = _pad_image(img, pad)
    output = np.zeros_like(img, dtype=np.float64)

    if img.ndim == 2:
        for i in range(h):
            for j in range(w):
                window = padded[i:i+k, j:j+k]
                output[i, j] = np.mean(window)
    else:
        for i in range(h):
            for j in range(w):
                for c in range(3):
                    window = padded[i:i+k, j:j+k, c]
                    output[i, j, c] = np.mean(window)

    # round untuk menjaga nilai integer visual lebih baik, lalu cast
    return np.clip(np.round(output), 0, 255).astype(np.uint8)

# Tampilkan Gambar helper
def show_image_grid(images, titles, cmap_flags):
    # images: list of arrays; same length as titles and cmap_flags
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title, cmap_flag in zip(axes, images, titles, cmap_flags):
        ax.imshow(img, cmap="gray" if cmap_flag else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Data loading (pakai path milikmu)
path_rgb1  = "c:/Users/ASUS/Documents/Project Pengcit 2/lanscape 1(pemandangan).jpg"
path_rgb2  = "c:/Users/ASUS/Documents/Project Pengcit 2/potrait 2(object).jpg"
path_gray1 = "c:/Users/ASUS/Documents/Project Pengcit 2/lanscape 1(grayscale).png" 
path_gray2 = "c:/Users/ASUS/Documents/Project Pengcit 2/potrait 2(grayscale).png"

img_rgb1  = np.array(Image.open(path_rgb1).convert("RGB"))
img_rgb2  = np.array(Image.open(path_rgb2).convert("RGB"))

# Pastikan grayscale benar-benar 2D (mode 'L')
img_gray1 = np.array(Image.open(path_gray1).convert("L"))
img_gray2 = np.array(Image.open(path_gray2).convert("L"))

# Level Derau
sp_levels = [0.02, 0.10]
gauss_levels = [(0, 10), (0, 30)]

# Menambah Noise ke 4 Citra
sp_rgb1_lvl1  = salt_pepper_noise(img_rgb1, sp_levels[0])
sp_rgb1_lvl2  = salt_pepper_noise(img_rgb1, sp_levels[1])
sp_rgb2_lvl1  = salt_pepper_noise(img_rgb2, sp_levels[0])
sp_rgb2_lvl2  = salt_pepper_noise(img_rgb2, sp_levels[1])

sp_gray1_lvl1 = salt_pepper_noise(img_gray1, sp_levels[0])
sp_gray1_lvl2 = salt_pepper_noise(img_gray1, sp_levels[1])
sp_gray2_lvl1 = salt_pepper_noise(img_gray2, sp_levels[0])
sp_gray2_lvl2 = salt_pepper_noise(img_gray2, sp_levels[1])

gauss_rgb1_lvl1  = gaussian_noise(img_rgb1, gauss_levels[0][0], gauss_levels[0][1])
gauss_rgb1_lvl2  = gaussian_noise(img_rgb1, gauss_levels[1][0], gauss_levels[1][1])
gauss_rgb2_lvl1  = gaussian_noise(img_rgb2, gauss_levels[0][0], gauss_levels[0][1])
gauss_rgb2_lvl2  = gaussian_noise(img_rgb2, gauss_levels[1][0], gauss_levels[1][1])

gauss_gray1_lvl1 = gaussian_noise(img_gray1, gauss_levels[0][0], gauss_levels[0][1])
gauss_gray1_lvl2 = gaussian_noise(img_gray1, gauss_levels[1][0], gauss_levels[1][1])
gauss_gray2_lvl1 = gaussian_noise(img_gray2, gauss_levels[0][0], gauss_levels[0][1])
gauss_gray2_lvl2 = gaussian_noise(img_gray2, gauss_levels[1][0], gauss_levels[1][1])

all_noisy_images = [
    sp_rgb1_lvl1, sp_rgb1_lvl2, sp_rgb2_lvl1, sp_rgb2_lvl2,
    sp_gray1_lvl1, sp_gray1_lvl2, sp_gray2_lvl1, sp_gray2_lvl2,
    gauss_rgb1_lvl1, gauss_rgb1_lvl2, gauss_rgb2_lvl1, gauss_rgb2_lvl2,
    gauss_gray1_lvl1, gauss_gray1_lvl2, gauss_gray2_lvl1, gauss_gray2_lvl2
]

# Terapkan filter dan simpan hasil
filtered_results = []
for img in all_noisy_images:
    min_f  = min_filter(img)
    max_f  = max_filter(img)
    med_f  = median_filter(img)
    mean_f = mean_filter(img)

    filtered_results.append({
        "noisy": img,
        "min": min_f,
        "max": max_f,
        "median": med_f,
        "mean": mean_f
    })

# Tampilkan setiap noisy image + 4 hasil filter (multi-figure: 1 figure per noisy image)
for idx, img in enumerate(all_noisy_images):
    print(f"Menampilkan hasil untuk citra noisy ke-{idx+1} ...")

    res = filtered_results[idx]
    images = [res["noisy"], res["min"], res["max"], res["median"], res["mean"]]
    titles = ["Noisy Image", "Min Filter", "Max Filter", "Median Filter", "Mean Filter"]
    cmap_flags = [im.ndim == 2 for im in images]  # True = grayscale

    show_image_grid(images, titles, cmap_flags)
