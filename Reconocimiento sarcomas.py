"""
Pipeline de preprocesamiento de radiografías de fémur para detección de patrones de osteosarcoma
- Carga masiva de imágenes (>500)
- Mejora de brillo/contraste con CLAHE
- Filtrado de ruido
- Detección de bordes
- Segmentación por Otsu
- Extracción básica de estadísticas para análisis posterior

Requiere:
    pip install opencv-python scikit-image pandas tqdm
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

# ==============================
# CONFIGURACIÓN
# ==============================
INPUT_DIR = Path("/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Sarcomas malignos")
OUTPUT_DIR = Path("/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Malignos")
SUBDIRS = ["enhanced", "denoised", "edges", "segmented"]

for sub in SUBDIRS:
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ==============================
# FUNCIONES DE PROCESAMIENTO
# ==============================
def load_grayscale(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar: {path}")
    return img


def enhance_contrast(img: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def denoise_image(img: np.ndarray):
    return cv2.GaussianBlur(img, (5, 5), 0)


def detect_edges(img: np.ndarray):
    return cv2.Canny(img, 50, 150)


def segment_otsu(img: np.ndarray):
    _, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def extract_basic_features(img: np.ndarray, mask: np.ndarray):
    roi = img[mask > 0]
    if roi.size == 0:
        return {
            "mean_intensity": 0,
            "std_intensity": 0,
            "area": 0,
            "contrast": 0,
            "homogeneity": 0,
        }

    # textura GLCM sobre imagen reducida para eficiencia
    resized = cv2.resize(img, (128, 128))
    glcm = graycomatrix(resized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    return {
        "mean_intensity": float(np.mean(roi)),
        "std_intensity": float(np.std(roi)),
        "area": int(np.sum(mask > 0)),
        "contrast": float(contrast),
        "homogeneity": float(homogeneity),
    }


# ==============================
# PIPELINE PRINCIPAL
# ==============================
def process_image(path: Path):
    img = load_grayscale(path)
    enhanced = enhance_contrast(img)
    denoised = denoise_image(enhanced)
    edges = detect_edges(denoised)
    segmented = segment_otsu(denoised)

    # guardar resultados intermedios
    cv2.imwrite(str(OUTPUT_DIR / "enhanced" / path.name), enhanced)
    cv2.imwrite(str(OUTPUT_DIR / "denoised" / path.name), denoised)
    cv2.imwrite(str(OUTPUT_DIR / "edges" / path.name), edges)
    cv2.imwrite(str(OUTPUT_DIR / "segmented" / path.name), segmented)

    features = extract_basic_features(denoised, segmented)
    features["image_name"] = path.name
    return features


def run_pipeline():
    image_paths = [p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in VALID_EXT]

    print(f"Total de imágenes encontradas: {len(image_paths)}")

    results = []
    for path in tqdm(image_paths, desc="Procesando radiografías"):
        try:
            results.append(process_image(path))
        except Exception as e:
            print(f"Error en {path.name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "features_estadisticas.csv", index=False)
    print("Pipeline completado. Features guardadas en CSV.")
    return df


if __name__ == "__main__":
    df_features = run_pipeline()
    print(df_features.head())
