import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# 1. Фіксоване визначення шляхiв
project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "results_practice_02"
results_dir.mkdir(parents=True, exist_ok=True)

# Генериратор для відтворюваності шуму
rng = np.random.default_rng(seed=42)


# 2. Функції додавання шуму
def add_gaussian_noise(image, sigma=20, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    noise = generator.normal(loc=0, scale=sigma, size=image.shape)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image, amount=0.05, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    noisy_image = image.copy()
    random_values = generator.random(image.shape)
    pepper_mask = random_values < amount / 2
    salt_mask = random_values > 1 - amount / 2
    noisy_image[pepper_mask] = 0
    noisy_image[salt_mask] = 255
    return noisy_image


def add_speckle_noise(image, sigma=0.20, generator=None):
    if generator is None:
        generator = np.random.default_rng()
    image_float = image.astype(np.float32)
    noise = generator.normal(loc=0, scale=sigma, size=image.shape)
    noisy_image = image_float + image_float * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# 3. Фільтрація та обчислення метрик
def apply_filters(image):
    return {
        "Mean": cv2.blur(image, (5, 5)),
        "Gaussian": cv2.GaussianBlur(image, (5, 5), sigmaX=1.5),
        "Median": cv2.medianBlur(image, 5),
        "Bilateral": cv2.bilateralFilter(
            image, d=9, sigmaColor=75, sigmaSpace=75
        ),
    }


def calculate_metrics(reference, processed):
    psnr = peak_signal_noise_ratio(reference, processed, data_range=255)
    ssim = structural_similarity(reference, processed, data_range=255)
    return psnr, ssim


def run_experiment_1(camera_image):
    gaussian_noisy = add_gaussian_noise(camera_image, sigma=20, generator=rng)
    salt_pepper_noisy = add_salt_and_pepper_noise(
        camera_image, amount=0.05, generator=rng
    )
    speckle_noisy = add_speckle_noise(camera_image, sigma=0.20, generator=rng)

    noise_cases = {
        "Gaussian noise": gaussian_noisy,
        "Salt-and-Pepper noise": salt_pepper_noisy,
        "Speckle noise": speckle_noisy,
    }

    filtering_results = {}
    metric_rows = []

    for noise_name, noisy_image in noise_cases.items():
        filters_out = apply_filters(noisy_image)
        filtering_results[noise_name] = filters_out

        noisy_psnr, noisy_ssim = calculate_metrics(camera_image, noisy_image)
        metric_rows.append(
            {
                "noise": noise_name,
                "method": "No filtering",
                "psnr": noisy_psnr,
                "ssim": noisy_ssim,
            }
        )

        for method_name, filtered_image in filters_out.items():
            psnr, ssim = calculate_metrics(camera_image, filtered_image)
            metric_rows.append(
                {
                    "noise": noise_name,
                    "method": method_name,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

    # Збереження метрик
    metrics_path = results_dir / "filtering_metrics.csv"
    with open(
        metrics_path, "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["noise", "method", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    # Колаж порівняння фільтрів
    column_names = [
        "Початкове",
        "Зашумлене",
        "Mean",
        "Gaussian",
        "Median",
        "Bilateral",
    ]
    fig, axes = plt.subplots(3, 6, figsize=(20, 11))
    for row_idx, (noise_name, noisy_img) in enumerate(noise_cases.items()):
        row_images = [
            camera_image,
            noisy_img,
            filtering_results[noise_name]["Mean"],
            filtering_results[noise_name]["Gaussian"],
            filtering_results[noise_name]["Median"],
            filtering_results[noise_name]["Bilateral"],
        ]
        for col_idx, (img, col_name) in enumerate(
            zip(row_images, column_names)
        ):
            axes[row_idx, col_idx].imshow(img, cmap="gray", vmin=0, vmax=255)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_name)
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(noise_name, fontsize=11)

    plt.tight_layout()
    plt.savefig(
        results_dir / "noise_filtering_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def run_experiment_2(page_image):
    global_threshold = 150
    _, page_global = cv2.threshold(
        page_image, global_threshold, 255, cv2.THRESH_BINARY_INV
    )

    otsu_thresh, page_otsu = cv2.threshold(
        page_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    page_adaptive = cv2.adaptiveThreshold(
        page_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=10,
    )

    thresholding_results = [
        (page_image, "Початкове зображення"),
        (page_global, f"Global, T={global_threshold}"),
        (page_otsu, f"Otsu, T={otsu_thresh:.0f}"),
        (page_adaptive, "Adaptive Gaussian"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for ax, (img, title) in zip(axes, thresholding_results):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        results_dir / "thresholding_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def run_experiment_3(coins_image):
    coins_blurred = cv2.GaussianBlur(coins_image, (5, 5), sigmaX=1.0)
    _, coins_binary = cv2.threshold(
        coins_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask_with_noise = coins_binary.copy()
    random_values = rng.random(coins_binary.shape)
    mask_with_noise[random_values < 0.01] = 255
    mask_with_noise[random_values > 0.99] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask_with_noise, kernel, iterations=1)
    dilated = cv2.dilate(mask_with_noise, kernel, iterations=1)
    opened = cv2.morphologyEx(mask_with_noise, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(mask_with_noise, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    morph_grad = cv2.morphologyEx(cleaned_mask, cv2.MORPH_GRADIENT, kernel)

    morph_results = [
        (coins_binary, "Початкова маска"),
        (mask_with_noise, "Маска з помилками"),
        (eroded, "Erosion"),
        (dilated, "Dilation"),
        (opened, "Opening"),
        (closed, "Closing"),
        (cleaned_mask, "Opening → Closing"),
        (morph_grad, "Morphological Gradient"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, (img, title) in zip(axes.ravel(), morph_results):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        results_dir / "morphology_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def run_standalone_task():
    # Самостійне завдання: аналіз мікроскопічного зображення клітин
    bio_image = data.human_mitosis()

    # 1. Додавання шуму Гаусса
    bio_noisy = add_gaussian_noise(bio_image, sigma=15, generator=rng)

    # 2. Фільтрація (Обрано Median Filter для збереження границь ядер)
    bio_filtered = cv2.medianBlur(bio_noisy, 5)

    # 3. Метрики
    psnr_noisy, ssim_noisy = calculate_metrics(bio_image, bio_noisy)
    psnr_filt, ssim_filt = calculate_metrics(bio_image, bio_filtered)

    # 4. Бінаризація Otsu (Клітини світліші за фон -> THRESH_BINARY)
    thresh_val, bio_binary = cv2.threshold(
        bio_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. Морфологічна обробка Opening -> Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bio_opened = cv2.morphologyEx(bio_binary, cv2.MORPH_OPEN, kernel)
    bio_cleaned = cv2.morphologyEx(bio_opened, cv2.MORPH_CLOSE, kernel)

    # 6. Побудова підсумкового колажу
    collage_items = [
        (bio_image, "1. Початкове зображення"),
        (bio_noisy, f"2. Gaussian Noise (σ=15)\nPSNR={psnr_noisy:.2f}dB"),
        (bio_filtered, f"3. Median Filter (5x5)\nPSNR={psnr_filt:.2f}dB"),
        (bio_binary, f"4. Otsu (T={thresh_val:.0f}, BINARY)"),
        (bio_opened, "5. Opening (Видалення шуму)"),
        (bio_cleaned, "6. Opening → Closing (Очищена маска)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (img, title) in zip(axes.ravel(), collage_items):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        "Конвеєр попередньої обробки біомедичного зображення клітин (human_mitosis)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        results_dir / "human_mitosis_preprocessing.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def main():
    camera_img = data.camera()
    page_img = data.page()
    coins_img = data.coins()

    # Збереження оглядового колажу
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (img, name) in zip(
        axes,
        [(camera_img, "Camera"), (page_img, "Page"), (coins_img, "Coins")],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\n{img.shape}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        results_dir / "input_images.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    run_experiment_1(camera_img)
    run_experiment_2(page_img)
    run_experiment_3(coins_img)
    run_standalone_task()

    print(
        "Результати збережено у results_practice_02/"
    )


if __name__ == "__main__":
    main()