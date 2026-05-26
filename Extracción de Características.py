"""
=============================================================================
SCRIPT 2 — EXTRACCIÓN DE CARACTERÍSTICAS (GLCM · HARALICK · ESTADÍSTICAS)
=============================================================================
Descripción:
    1. Carga las imágenes ORIGINALES y las SEGMENTADAS (salida del Script 1).
    2. Realiza la multiplicación 1‑a‑1 (original × máscara normalizada).
    3. Extrae características sobre la región de interés (ROI):
         • GLCM  : energía, entropía, contraste, correlación, homogeneidad,
                   disimilaridad, ASM
         • Haralick (mahotas): 13 descriptores de Haralick
         • Estadísticas de primer orden: media, mediana, std, varianza,
           kurtosis, skewness, percentiles 25/75, rango intercuartílico,
           entropía de Shannon, energía, mínimo, máximo, rango
         • Morfología del contorno: área, perímetro, circularidad,
           solidez, excentricidad (vía Active Contour / Snakes)
    4. Exporta un CSV con todas las características etiquetadas
       (0 = benigno, 1 = maligno).

Estructura esperada de entrada:
    dataset/
    ├── Sarcomas Benignos/   ← imágenes originales
    └── Sarcomas Malignos/

    output_segmentacion/
    ├── Sarcomas Benignos/   ← *_segmentada.png  y  *_mascara.png
    └── Sarcomas Malignos/

Salida:
    output_caracteristicas/
    └── caracteristicas.csv

Librerías requeridas:
    pip install opencv-python numpy pandas matplotlib scikit-image
                scipy mahotas tqdm
=============================================================================
"""

import os
import warnings
import cv2
import numpy as np
import pandas as pd
import mahotas
from pathlib import Path
from scipy import stats
from skimage import measure, filters, exposure
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
DATASET_ROOT      = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/dataset 1"
SEG_ROOT          = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Segmentación"
OUTPUT_ROOT       = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Características"
CATEGORIAS        = {
    "Sarcomas Benignoss": 0,
    "Sarcomas Malignos": 1,
}
EXTENSIONES       = (".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp")

# Parámetros GLCM
GLCM_DISTANCIAS   = [1, 2, 3]
GLCM_ANGULOS      = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_NIVELES      = 256

# Parámetros Active Contour (Snakes) para morfología
SNAKE_ALPHA       = 0.015   # Rigidez de la curva
SNAKE_BETA        = 10      # Suavizado de la curva
SNAKE_GAMMA       = 0.001   # Paso temporal
SNAKE_ITER        = 2500    # Iteraciones máximas
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════

def cargar_par(ruta_original: str, ruta_segmentada: str):
    """Carga la imagen original y la segmentada; devuelve ambas en gris."""
    orig = cv2.imread(ruta_original, cv2.IMREAD_GRAYSCALE)
    seg  = cv2.imread(ruta_segmentada, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"Original no encontrada: {ruta_original}")
    if seg is None:
        raise FileNotFoundError(f"Segmentada no encontrada: {ruta_segmentada}")
    # Asegurar mismo tamaño
    if orig.shape != seg.shape:
        seg = cv2.resize(seg, (orig.shape[1], orig.shape[0]))
    return orig, seg


def multiplicar_imagenes(orig: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Multiplicación 1‑a‑1: original × (segmentada / 255).
    Retorna la ROI con fondo negro.
    """
    mascara_norm = seg.astype(np.float64) / 255.0
    roi = (orig.astype(np.float64) * mascara_norm).astype(np.uint8)
    return roi


def extraer_roi_pixeles(roi: np.ndarray) -> np.ndarray:
    """Retorna sólo los píxeles no nulos de la ROI."""
    return roi[roi > 0].astype(np.float64)


# ══════════════════════════════════════════════════════════════
#  CARACTERÍSTICAS GLCM
# ══════════════════════════════════════════════════════════════

def caracteristicas_glcm(roi: np.ndarray) -> dict:
    """
    Calcula propiedades GLCM promediadas sobre distancias y ángulos.
    Propiedades: energía, entropía, contraste, correlación,
                 homogeneidad, disimilaridad, ASM.
    """
    # Recortar ROI a 8 bits y normalizar niveles
    roi_8 = np.clip(roi, 0, 255).astype(np.uint8)

    glcm = graycomatrix(
        roi_8,
        distances=GLCM_DISTANCIAS,
        angles=GLCM_ANGULOS,
        levels=GLCM_NIVELES,
        symmetric=True,
        normed=True,
    )

    propiedades = ["energy", "contrast", "correlation",
                   "homogeneity", "dissimilarity", "ASM"]
    resultado = {}
    for prop in propiedades:
        val = graycoprops(glcm, prop)  # shape (n_dist, n_ang)
        resultado[f"glcm_{prop}"] = float(np.mean(val))

    # Entropía de la GLCM (no disponible en graycoprops, se calcula manual)
    glcm_sum = glcm.sum(axis=(2, 3), keepdims=True)
    glcm_norm = glcm / (glcm_sum + 1e-10)
    entropia_glcm = -np.sum(
        glcm_norm * np.log2(glcm_norm + 1e-10)
    ) / (len(GLCM_DISTANCIAS) * len(GLCM_ANGULOS))
    resultado["glcm_entropia"] = float(entropia_glcm)

    return resultado


# ══════════════════════════════════════════════════════════════
#  CARACTERÍSTICAS HARALICK (mahotas)
# ══════════════════════════════════════════════════════════════

HARALICK_NOMBRES = [
    "har_asm", "har_contraste", "har_correlacion",
    "har_varianza_suma", "har_idm", "har_promedio_suma",
    "har_varianza_diferencia", "har_entropia_suma",
    "har_entropia", "har_entropia_diferencia",
    "har_coef_correlacion1", "har_coef_correlacion2",
    "har_max_prob_correlacion",
]

def caracteristicas_haralick(roi: np.ndarray) -> dict:
    """Calcula los 13 descriptores de Haralick usando mahotas."""
    roi_8 = np.clip(roi, 0, 255).astype(np.uint8)
    try:
        har = mahotas.features.haralick(roi_8, ignore_zeros=True,
                                         return_mean=True)
    except Exception:
        har = np.zeros(13)
    return {nombre: float(val) for nombre, val in zip(HARALICK_NOMBRES, har)}


# ══════════════════════════════════════════════════════════════
#  ESTADÍSTICAS DE PRIMER ORDEN
# ══════════════════════════════════════════════════════════════

def estadisticas_primer_orden(pixeles: np.ndarray) -> dict:
    """
    Calcula estadísticas de primer orden sobre los píxeles de la ROI.
    """
    if len(pixeles) == 0:
        pixeles = np.array([0.0])

    p25, p75 = np.percentile(pixeles, [25, 75])
    # Entropía de Shannon
    hist, _ = np.histogram(pixeles, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    shannon = -np.sum(hist * np.log2(hist + 1e-10))

    return {
        "stat_media":         float(np.mean(pixeles)),
        "stat_mediana":       float(np.median(pixeles)),
        "stat_std":           float(np.std(pixeles)),
        "stat_varianza":      float(np.var(pixeles)),
        "stat_kurtosis":      float(stats.kurtosis(pixeles)),
        "stat_skewness":      float(stats.skew(pixeles)),
        "stat_p25":           float(p25),
        "stat_p75":           float(p75),
        "stat_iqr":           float(p75 - p25),
        "stat_entropia_sh":   float(shannon),
        "stat_energia":       float(np.sum(pixeles ** 2) / len(pixeles)),
        "stat_minimo":        float(np.min(pixeles)),
        "stat_maximo":        float(np.max(pixeles)),
        "stat_rango":         float(np.max(pixeles) - np.min(pixeles)),
        "stat_n_pixeles":     int(len(pixeles)),
    }


# ══════════════════════════════════════════════════════════════
#  MORFOLOGÍA DEL CONTORNO (Active Contour / Snakes)
# ══════════════════════════════════════════════════════════════

def morfologia_contorno(roi: np.ndarray) -> dict:
    """
    Ajusta un contorno activo (Snake) a la región de interés y extrae
    métricas morfológicas: área, perímetro, circularidad, solidez,
    excentricidad.
    """
    resultado = {
        "morfo_area": 0.0, "morfo_perimetro": 0.0,
        "morfo_circularidad": 0.0, "morfo_solidez": 0.0,
        "morfo_excentricidad": 0.0,
    }

    mascara_bin = (roi > 0).astype(np.uint8)
    if mascara_bin.sum() < 100:
        return resultado

    # Obtener contorno externo para inicializar el snake
    contornos, _ = cv2.findContours(mascara_bin, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return resultado

    cont_principal = max(contornos, key=cv2.contourArea)
    if len(cont_principal) < 10:
        return resultado

    # Crear contorno de inicialización (elipse sobre bbox del contorno)
    x, y, w, h = cv2.boundingRect(cont_principal)
    s = np.linspace(0, 2 * np.pi, 300)
    cx, cy = x + w / 2, y + h / 2
    init = np.column_stack([
        cy + (h / 2) * np.sin(s),
        cx + (w / 2) * np.cos(s),
    ])

    # Normalizar imagen para snake
    img_norm = exposure.rescale_intensity(roi.astype(np.float64),
                                          out_range=(0, 1))
    img_suave = gaussian(img_norm, sigma=3)

    try:
        snake = active_contour(
            img_suave, init,
            alpha=SNAKE_ALPHA, beta=SNAKE_BETA,
            gamma=SNAKE_GAMMA, max_num_iter=SNAKE_ITER,
            coordinates="rc",
        )

        # Crear máscara desde el snake para medir propiedades
        mascara_snake = np.zeros(roi.shape, dtype=np.uint8)
        puntos_snake = snake[:, ::-1].astype(np.int32)
        cv2.fillPoly(mascara_snake, [puntos_snake], 255)

        etiquetas = measure.label(mascara_snake > 0)
        props = measure.regionprops(etiquetas)
        if props:
            r = max(props, key=lambda x: x.area)
            area = r.area
            perim = r.perimeter if r.perimeter > 0 else 1
            circ = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
            resultado.update({
                "morfo_area":          float(area),
                "morfo_perimetro":     float(perim),
                "morfo_circularidad":  float(np.clip(circ, 0, 1)),
                "morfo_solidez":       float(r.solidity),
                "morfo_excentricidad": float(r.eccentricity),
            })
    except Exception:
        # Si el snake falla, usar propiedades directas de la máscara
        etiquetas = measure.label(mascara_bin)
        props = measure.regionprops(etiquetas)
        if props:
            r = max(props, key=lambda x: x.area)
            perim = r.perimeter if r.perimeter > 0 else 1
            circ = (4 * np.pi * r.area) / (perim ** 2)
            resultado.update({
                "morfo_area":          float(r.area),
                "morfo_perimetro":     float(perim),
                "morfo_circularidad":  float(np.clip(circ, 0, 1)),
                "morfo_solidez":       float(r.solidity),
                "morfo_excentricidad": float(r.eccentricity),
            })

    return resultado


# ══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def extraer_caracteristicas_imagen(ruta_orig: str, ruta_seg: str) -> dict:
    """Pipeline completo de extracción para una imagen."""
    orig, seg = cargar_par(ruta_orig, ruta_seg)
    roi       = multiplicar_imagenes(orig, seg)
    pixeles   = extraer_roi_pixeles(roi)

    caracteristicas = {}
    caracteristicas.update(caracteristicas_glcm(roi))
    caracteristicas.update(caracteristicas_haralick(roi))
    caracteristicas.update(estadisticas_primer_orden(pixeles))
    caracteristicas.update(morfologia_contorno(roi))

    return caracteristicas


def procesar_dataset():
    print("=" * 60)
    print("  EXTRACCIÓN DE CARACTERÍSTICAS — SARCOMAS ÓSEOS")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    registros = []

    for categoria, etiqueta in CATEGORIAS.items():
        carpeta_orig = os.path.join(DATASET_ROOT, categoria)
        carpeta_seg  = os.path.join(SEG_ROOT, categoria)

        if not os.path.isdir(carpeta_orig):
            print(f"\n⚠  No existe: {carpeta_orig} — se omite.")
            continue

        imagenes = [
            f for f in os.listdir(carpeta_orig)
            if f.lower().endswith(EXTENSIONES)
        ]

        print(f"\n📁 {categoria}  ({len(imagenes)} imágenes)  |  etiqueta = {etiqueta}")

        for nombre_archivo in tqdm(imagenes, desc=f"  {categoria[:22]}", unit="img"):
            nombre_base = Path(nombre_archivo).stem
            ruta_orig = os.path.join(carpeta_orig, nombre_archivo)
            ruta_seg  = os.path.join(carpeta_seg, f"{nombre_base}_segmentada.png")

            if not os.path.isfile(ruta_seg):
                print(f"\n  ⚠ Segmentada no encontrada para {nombre_archivo}, se omite.")
                continue

            try:
                caract = extraer_caracteristicas_imagen(ruta_orig, ruta_seg)
                caract["imagen"]    = nombre_base
                caract["categoria"] = categoria
                caract["etiqueta"]  = etiqueta
                registros.append(caract)
            except Exception as e:
                print(f"\n  ✗ Error en {nombre_archivo}: {e}")

    if not registros:
        print("\n⚠ No se extrajeron características. Verifica las rutas.")
        return None

    df = pd.DataFrame(registros)
    # Reordenar columnas: metadata al frente
    cols_meta = ["imagen", "categoria", "etiqueta"]
    cols_feat = [c for c in df.columns if c not in cols_meta]
    df = df[cols_meta + cols_feat]

    ruta_csv = os.path.join(OUTPUT_ROOT, "caracteristicas.csv")
    df.to_csv(ruta_csv, index=False)

    print("\n" + "=" * 60)
    print(f"  ✅ Total de imágenes procesadas: {len(df)}")
    print(f"  📊 Características extraídas:    {len(cols_feat)}")
    print(f"  📄 CSV guardado en: {os.path.abspath(ruta_csv)}")
    print("=" * 60)

    # Resumen estadístico por clase
    print("\n📋 Resumen por clase:")
    print(df.groupby("categoria")[cols_feat[:6]].mean().round(4).to_string())

    return df


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = procesar_dataset()
    if df is not None:
        print(f"\n📌 DataFrame shape: {df.shape}")
        print(df.head(3).to_string())
