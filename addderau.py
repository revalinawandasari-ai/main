import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fungsi Salt & Pepper Noise (manual)
def salt_pepper_noise(img, prob):
    """
    Menambahkan derau salt & pepper secara manual.
    img  : numpy array (RGB atau grayscale)
    prob : probabilitas noise (0.02, 0.10)
    """

    noisy = img.copy()
    h, w = img.shape[:2]

    # Matriks acak bernilai 0-1
    rand = np.random.rand(h, w)

    # Salt (putih)
    noisy[rand < prob / 2] = 255

    # Pepper (hitam)
    noisy[rand > 1 - (prob / 2)] = 0

    return noisy

# Fungsi Gaussian Noise (manual)
def gaussian_noise(img, mean, sigma):
    """
    Menambahkan derau Gaussian secara manual.
    mean  : nilai rata-rata (biasanya 0)
    sigma : standar deviasi (semakin besar → noise makin tinggi)
    """

    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(float) + noise

    # Batasi 0–255
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)

# Tampilkan Gambar
def show_image(img, title, cmap=None):
    plt.figure(figsize=(5, 4))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Masukkan Path 4 Gambar (2 RGB, 2 Grayscale)
path_rgb1  = "c:/Users/ASUS/Documents/Project Pengcit 2/lanscape 1(pemandangan).jpg"
path_rgb2  = "c:/Users/ASUS/Documents/Project Pengcit 2/potrait 2(object).jpg"
path_gray1 = "c:/Users/ASUS/Documents/Project Pengcit 2/lanscape 1(grayscale).png" 
path_gray2 = "c:/Users/ASUS/Documents/Project Pengcit 2/potrait 2(grayscale).png"

# Membaca Citra
img_rgb1  = np.array(Image.open(path_rgb1))
img_rgb2  = np.array(Image.open(path_rgb2))
img_gray1 = np.array(Image.open(path_gray1))
img_gray2 = np.array(Image.open(path_gray2))

# Level Derau
sp_levels = [0.02, 0.10]         # Salt & Pepper 2 tingkat
gauss_levels = [(0, 10), (0, 30)]  # Gaussian 2 tingkat (mean=0)

# Menambah Noise ke 4 Citra

# Salt & Pepper
sp_rgb1_lvl1  = salt_pepper_noise(img_rgb1, sp_levels[0])
sp_rgb1_lvl2  = salt_pepper_noise(img_rgb1, sp_levels[1])
sp_rgb2_lvl1  = salt_pepper_noise(img_rgb2, sp_levels[0])
sp_rgb2_lvl2  = salt_pepper_noise(img_rgb2, sp_levels[1])

sp_gray1_lvl1 = salt_pepper_noise(img_gray1, sp_levels[0])
sp_gray1_lvl2 = salt_pepper_noise(img_gray1, sp_levels[1])
sp_gray2_lvl1 = salt_pepper_noise(img_gray2, sp_levels[0])
sp_gray2_lvl2 = salt_pepper_noise(img_gray2, sp_levels[1])

# Gaussian
gauss_rgb1_lvl1  = gaussian_noise(img_rgb1, gauss_levels[0][0], gauss_levels[0][1])
gauss_rgb1_lvl2  = gaussian_noise(img_rgb1, gauss_levels[1][0], gauss_levels[1][1])
gauss_rgb2_lvl1  = gaussian_noise(img_rgb2, gauss_levels[0][0], gauss_levels[0][1])
gauss_rgb2_lvl2  = gaussian_noise(img_rgb2, gauss_levels[1][0], gauss_levels[1][1])

gauss_gray1_lvl1 = gaussian_noise(img_gray1, gauss_levels[0][0], gauss_levels[0][1])
gauss_gray1_lvl2 = gaussian_noise(img_gray1, gauss_levels[1][0], gauss_levels[1][1])
gauss_gray2_lvl1 = gaussian_noise(img_gray2, gauss_levels[0][0], gauss_levels[0][1])
gauss_gray2_lvl2 = gaussian_noise(img_gray2, gauss_levels[1][0], gauss_levels[1][1])

# Menampilkan Seluruh Gambar
show_image(sp_rgb1_lvl1, "Salt & Pepper Level 1 (0.02) — RGB 1")
show_image(sp_rgb1_lvl2, "Salt & Pepper Level 2 (0.10) — RGB 1")
show_image(sp_rgb2_lvl1, "Salt & Pepper Level 1 (0.02) — RGB 2")
show_image(sp_rgb2_lvl2, "Salt & Pepper Level 2 (0.10) — RGB 2")

show_image(sp_gray1_lvl1, "Salt & Pepper Level 1 (0.02)  — Grayscale 1", cmap="gray")
show_image(sp_gray1_lvl2, "Salt & Pepper Level 2 (0.10) — Grayscale 1", cmap="gray")
show_image(sp_gray2_lvl1, "Salt & Pepper Level 1 (0.02) — Grayscale 2", cmap="gray")
show_image(sp_gray2_lvl2, "Salt & Pepper Level 2 (0.10) — Grayscale 2", cmap="gray")

show_image(gauss_rgb1_lvl1, "Gaussian Level 1 (0, 10) — RGB 1")
show_image(gauss_rgb1_lvl2, "Gaussian Level 2 (0, 30) — RGB 1")
show_image(gauss_rgb2_lvl1, "Gaussian Level 1 (0, 10) — RGB 2")
show_image(gauss_rgb2_lvl2, "Gaussian Level 2 (0, 30) — RGB 2")

show_image(gauss_gray1_lvl1, "Gaussian Level 1 (0, 10) — Grayscale 1", cmap="gray")
show_image(gauss_gray1_lvl2, "Gaussian Level 2 (0, 30) — Grayscale 1", cmap="gray")
show_image(gauss_gray2_lvl1, "Gaussian Level 1 (0, 10) — Grayscale 2", cmap="gray")
show_image(gauss_gray2_lvl2, "Gaussian Level 2 (0, 30) — Grayscale 2", cmap="gray")
