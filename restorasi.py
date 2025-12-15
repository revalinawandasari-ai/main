import numpy as np
from PIL import Image

# =====================================================
# MSE MANUAL
# =====================================================
def mse(img_ref, img_test):
    return np.mean((img_ref.astype(float) - img_test.astype(float)) ** 2)

# =====================================================
# RGB → GRAYSCALE (MANUAL)
# =====================================================
def rgb_to_grayscale_manual(img):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.299*R + 0.587*G + 0.114*B
    return gray.astype(np.uint8)

# =====================================================
# NOISE
# =====================================================
def salt_pepper_noise(img, prob):
    noisy = img.copy()
    h, w = img.shape[:2]
    rand = np.random.rand(h, w)

    salt = rand < prob/2
    pepper = rand > 1 - prob/2

    if img.ndim == 2:
        noisy[salt] = 255
        noisy[pepper] = 0
    else:
        noisy[salt,:] = 255
        noisy[pepper,:] = 0
    return noisy

def gaussian_noise(img, mean, sigma):
    noise = np.random.normal(mean, sigma, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

# =====================================================
# FILTER MANUAL (3x3 agar cepat & stabil)
# =====================================================
def pad(img):
    return np.pad(img, 1, mode="edge")

def min_filter(img):
    h, w = img.shape
    p = pad(img)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            out[i,j] = np.min(p[i:i+3, j:j+3])
    return out

def max_filter(img):
    h, w = img.shape
    p = pad(img)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            out[i,j] = np.max(p[i:i+3, j:j+3])
    return out

def median_filter(img):
    h, w = img.shape
    p = pad(img)
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            out[i,j] = np.median(p[i:i+3, j:j+3])
    return out

def mean_filter(img):
    h, w = img.shape
    p = pad(img)
    out = np.zeros_like(img, dtype=float)
    for i in range(h):
        for j in range(w):
            out[i,j] = np.mean(p[i:i+3, j:j+3])
    return np.round(out).astype(np.uint8)

# =====================================================
# LOAD IMAGE
# =====================================================
rgb_land = np.array(Image.open(
    "c:/Users/ASUS/Documents/Project Pengcit 2/lanscape 1(pemandangan).jpg"
).convert("RGB"))

rgb_port = np.array(Image.open(
    "c:/Users/ASUS/Documents/Project Pengcit 2/potrait 2(object).jpg"
).convert("RGB"))

gray_land = rgb_to_grayscale_manual(rgb_land)
gray_port = rgb_to_grayscale_manual(rgb_port)

datasets = [
    ("RGB_Landscape", rgb_land),
    ("RGB_Portrait", rgb_port),
    ("Gray_Landscape", gray_land),
    ("Gray_Portrait", gray_port)
]

# =====================================================
# PROSES & MSE
# =====================================================
print("\n=========== HASIL MSE RESTORASI ===========")

for name, img in datasets:
    print(f"\n--- {name} ---")

    for p in [0.02, 0.10]:
        noisy = salt_pepper_noise(img, p)
        print(f"S&P {p} | Noisy MSE = {mse(img, noisy):.4f}")

        if img.ndim == 2:
            print(f"   Min    = {mse(img, min_filter(noisy)):.4f}")
            print(f"   Max    = {mse(img, max_filter(noisy)):.4f}")
            print(f"   Median = {mse(img, median_filter(noisy)):.4f}")
            print(f"   Mean   = {mse(img, mean_filter(noisy)):.4f}")
        else:
            print("   (Filtering RGB dilewati)")

    for s in [10, 30]:
        noisy = gaussian_noise(img, 0, s)
        print(f"Gaussian σ={s} | Noisy MSE = {mse(img, noisy):.4f}")

        if img.ndim == 2:
            print(f"   Min    = {mse(img, min_filter(noisy)):.4f}")
            print(f"   Max    = {mse(img, max_filter(noisy)):.4f}")
            print(f"   Median = {mse(img, median_filter(noisy)):.4f}")
            print(f"   Mean   = {mse(img, mean_filter(noisy)):.4f}")
        else:
            print("   (Filtering RGB dilewati)")