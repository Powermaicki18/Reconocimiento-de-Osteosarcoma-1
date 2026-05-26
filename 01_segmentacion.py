"""
=============================================================================
SCRIPT 1 — SEGMENTACIÓN DE RADIOGRAFÍAS DE SARCOMAS ÓSEOS
        (con relleno de huecos por morfología matemática)
=============================================================================
Descripción:
    Carga imágenes desde las carpetas /Sarcomas Benignos y /Sarcomas Malignos,
    aplica segmentación automática (umbralización + morfología) para aislar
    el hueso, guarda las máscaras e imágenes segmentadas, y muestra resultados.

    Pipeline de relleno de huecos (en orden de aplicación):
      1. Cierre morfológico  — une grietas y bordes cercanos con un kernel elíptico
      2. Flood-fill inverso  — rellena todo lo que NO es fondo desde las esquinas
      3. fillConvexHull      — rellena la envoltura convexa de cada región
      4. Binary closing (skimage) con disco grande — cierra huecos residuales
    Cada paso actúa sobre el residuo del anterior, lo que permite cerrar huecos
    de distintos tamaños y formas sin distorsionar el contorno externo del hueso.

Estructura esperada del dataset:
    dataset/
    ├── Sarcomas Benignos/
    │   ├── IMG000006.jpeg
    │   └── ...
    └── Sarcomas Malignos/
        ├── IMG000012.jpeg
        └── ...

Salida:
    output_segmentacion/
    ├── Sarcomas Benignos/
    │   ├── IMG000006_segmentada.png
    │   ├── IMG000006_mascara.png
    │   └── ...
    └── Sarcomas Malignos/
        ├── IMG000012_segmentada.png
        ├── IMG000012_mascara.png
        └── ...

Librerías requeridas:
    pip install opencv-python numpy matplotlib scikit-image tqdm scipy
=============================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import morphology, filters, measure
from skimage.filters import gaussian
from skimage.morphology import disk, binary_closing
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN — Ajusta estas rutas antes de ejecutar
# ─────────────────────────────────────────────────────────────
DATASET_ROOT   = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/dataset" # Carpeta raíz del dataset
OUTPUT_ROOT    = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Segmentación 2" # Carpeta de salida
CATEGORIAS     = ["Sarcomas Benignos ", "Sarcomas Malignos"]
EXTENSIONES    = (".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp")

# Parámetros de segmentación
CLAHE_CLIP      = 3.0    # Límite de contraste para CLAHE
CLAHE_GRID      = (8, 8) # Tamaño de la cuadrícula CLAHE
UMBRAL_OFFSET   = 0.05   # Offset sobre el umbral de Otsu (-1 a 1)
MIN_AREA_RATIO  = 0.01   # Área mínima de región como fracción de la imagen
MAX_AREAS       = 3      # Máximo de regiones a conservar (la más grande)
SIGMA_GAUSS     = 2.0    # Suavizado gaussiano previo a la segmentación

# Parámetros de morfología matemática para relleno de huecos
MORFO_CIERRE_RADIO      = 15   # Radio del kernel de cierre inicial (píxeles)
MORFO_CIERRE_ITER       = 3    # Iteraciones del cierre inicial
MORFO_APERTURA_RADIO    = 5    # Radio del kernel de apertura (elimina ruido)
MORFO_APERTURA_ITER     = 2    # Iteraciones de la apertura
MORFO_DISCO_GRANDE      = 25   # Radio del disco para cierre final (huecos grandes)
MORFO_USAR_CONVEX_HULL  = True # Rellenar envoltura convexa por región
# ─────────────────────────────────────────────────────────────


def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga imagen en escala de grises y normaliza a [0, 255] uint8."""
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")
    return img


def mejorar_contraste(img: np.ndarray) -> np.ndarray:
    """Aplica CLAHE para realzar el contraste local de la radiografía."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(img)


def rellenar_huecos_pipeline(mascara: np.ndarray) -> np.ndarray:
    """
    Pipeline escalonado de relleno de huecos por morfología matemática.
    Aplica 4 técnicas en orden, de menos a más agresiva, para cubrir
    huecos de distintos tamaños y formas sin distorsionar el contorno externo.

    Etapa 1 — Cierre morfológico (cv2.MORPH_CLOSE)
        Une grietas finas y pequeños huecos en el borde.
        Kernel elíptico de radio MORFO_CIERRE_RADIO.

    Etapa 2 — Flood-fill inverso (floodFill desde esquinas)
        Identifica el fondo verdadero (conectado al borde de la imagen)
        y rellena todo lo interior que no pertenece al fondo.
        Cierra huecos de cualquier tamaño que no toquen el borde.

    Etapa 3 — scipy.ndimage.binary_fill_holes
        Relleno topológico: cierra cualquier componente conexa de fondo
        completamente rodeada por primer plano. Complementa el flood-fill
        en casos de morfología compleja.

    Etapa 4 — Envoltura convexa por región (opcional, MORFO_USAR_CONVEX_HULL)
        Para cada región etiquetada calcula su convex hull y lo rellena.
        Ideal para huesos con concavidades profundas o bordes irregulares.

    Etapa 5 — Cierre con disco grande (skimage.morphology.binary_closing)
        Cierre final con un disco de radio MORFO_DISCO_GRANDE para sellar
        cualquier hueco residual de mayor tamaño que no cerraron las etapas anteriores.

    Returns
    -------
    mascara : np.ndarray uint8
        Máscara binaria con huecos rellenados (valores 0 / 255).
    """
    # ── Etapa 1: Cierre morfológico inicial ──────────────────────────────
    kernel_cierre = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORFO_CIERRE_RADIO * 2 + 1, MORFO_CIERRE_RADIO * 2 + 1)
    )
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel_cierre,
        iterations=MORFO_CIERRE_ITER
    )

    # ── Etapa 2: Flood-fill inverso desde las 4 esquinas ─────────────────
    # Invertir: fondo → blanco, objeto → negro
    mascara_inv = cv2.bitwise_not(mascara)
    h, w = mascara_inv.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Sembrar desde esquina superior-izquierda (punto de fondo seguro)
    cv2.floodFill(mascara_inv, flood_mask, (0, 0), 255)
    # Los píxeles que aún son negros en mascara_inv son huecos interiores
    huecos_interiores = cv2.bitwise_not(mascara_inv)
    mascara = cv2.bitwise_or(mascara, huecos_interiores)

    # ── Etapa 3: Relleno topológico con scipy ────────────────────────────
    mascara_bool = mascara > 0
    mascara_bool = binary_fill_holes(mascara_bool)
    mascara = (mascara_bool.astype(np.uint8)) * 255

    return mascara


def segmentar_hueso(img: np.ndarray) -> np.ndarray:
    """
    Segmenta la región ósea usando:
      1. Suavizado gaussiano
      2. Umbralización de Otsu con offset ajustable
      3. Apertura morfológica (elimina ruido fino)
      4. Pipeline de relleno de huecos (5 etapas)
      5. Selección de las regiones más grandes (hueso principal)
    Devuelve una máscara binaria uint8 (0 / 255).
    """
    # 1. Suavizado
    img_suave = gaussian(img, sigma=SIGMA_GAUSS, preserve_range=True).astype(np.uint8)

    # 2. Umbralización de Otsu
    umbral_otsu  = filters.threshold_otsu(img_suave)
    umbral_final = umbral_otsu * (1 + UMBRAL_OFFSET)
    mascara = (img_suave > umbral_final).astype(np.uint8) * 255

    # 3. Apertura para eliminar ruido fino antes del relleno
    kernel_apertura = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORFO_APERTURA_RADIO * 2 + 1, MORFO_APERTURA_RADIO * 2 + 1)
    )
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel_apertura,
        iterations=MORFO_APERTURA_ITER
    )

    # # 4. Pipeline de relleno de huecos (5 etapas morfológicas)
    # mascara = rellenar_huecos_pipeline(mascara)
    #
    # # 5. Conservar únicamente las regiones más grandes
    # etiquetas    = measure.label(mascara > 0)
    # propiedades  = measure.regionprops(etiquetas)
    # area_imagen  = img.shape[0] * img.shape[1]
    # area_min     = area_imagen * MIN_AREA_RATIO
    #
    # regiones_validas = sorted(
    #     [r for r in propiedades if r.area >= area_min],
    #     key=lambda x: x.area, reverse=True
    # )[:MAX_AREAS]
    #
    # mascara_final = np.zeros_like(mascara)
    # for region in regiones_validas:
    #     mascara_final[etiquetas == region.label] = 255

    return mascara


def aplicar_mascara(img_original: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Multiplica la imagen original por la máscara normalizada para obtener la imagen segmentada."""
    mascara_norm = mascara / 255.0
    segmentada = (img_original * mascara_norm).astype(np.uint8)
    return segmentada


def guardar_resultados(nombre_base: str, img_original: np.ndarray,
                       mascara: np.ndarray, segmentada: np.ndarray,
                       carpeta_salida: str) -> dict:
    """Guarda máscara e imagen segmentada. Retorna rutas guardadas."""
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_mascara    = os.path.join(carpeta_salida, f"{nombre_base}_mascara.png")
    ruta_segmentada = os.path.join(carpeta_salida, f"{nombre_base}_segmentada.png")

    cv2.imwrite(ruta_mascara,    mascara)
    cv2.imwrite(ruta_segmentada, segmentada)

    return {"mascara": ruta_mascara, "segmentada": ruta_segmentada}


def visualizar_muestra(muestras: list, titulo: str = "Resultados de Segmentación"):
    """Muestra una grilla con ejemplos: original | máscara antes | máscara rellena | segmentada."""
    n = len(muestras)
    if n == 0:
        print("⚠ No hay muestras para visualizar.")
        return

    tiene_antes = "mascara_antes" in muestras[0]
    ncols = 4 if tiene_antes else 3
    columnas = (["Original", "Máscara (sin rellenar)", "Máscara (rellena)", "Segmentada final"]
                if tiene_antes else
                ["Original", "Máscara", "Segmentada"])
    claves   = (["original", "mascara_antes", "mascara", "segmentada"]
                if tiene_antes else
                ["original", "mascara", "segmentada"])

    fig = plt.figure(figsize=(ncols * 4, 5 * n), facecolor="#0d1117")
    fig.suptitle(titulo, fontsize=16, color="white", y=1.01)

    gs = gridspec.GridSpec(n, ncols, figure=fig, hspace=0.4, wspace=0.15)

    for i, muestra in enumerate(muestras):
        for j, (clave, col_titulo) in enumerate(zip(claves, columnas)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(muestra[clave], cmap="gray")
            ax.set_title(f"{col_titulo}\n{muestra['nombre']}", color="white",
                         fontsize=8, pad=4)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

    plt.savefig(os.path.join(OUTPUT_ROOT, "muestra_segmentacion.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
    print(f"\n📊 Figura guardada en: {os.path.join(OUTPUT_ROOT, 'muestra_segmentacion.png')}")


def procesar_dataset():
    """Procesa todas las imágenes del dataset y guarda los resultados."""
    print("=" * 60)
    print("  SEGMENTACIÓN DE RADIOGRAFÍAS — SARCOMAS ÓSEOS")
    print("=" * 60)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    total_procesadas = 0
    muestras_viz = []   # Guardar 1 muestra por categoría para visualización

    for categoria in CATEGORIAS:
        carpeta_entrada = os.path.join(DATASET_ROOT, categoria)
        carpeta_salida  = os.path.join(OUTPUT_ROOT, categoria)

        if not os.path.isdir(carpeta_entrada):
            print(f"\n⚠  Carpeta no encontrada: {carpeta_entrada} — se omite.")
            continue

        # Listar imágenes válidas
        imagenes = [
            f for f in os.listdir(carpeta_entrada)
            if f.lower().endswith(EXTENSIONES)
        ]

        if not imagenes:
            print(f"\n⚠  No se encontraron imágenes en: {carpeta_entrada}")
            continue

        print(f"\n📁 Procesando: {categoria}  ({len(imagenes)} imágenes)")
        os.makedirs(carpeta_salida, exist_ok=True)

        for nombre_archivo in tqdm(imagenes, desc=f"  {categoria[:25]}", unit="img"):
            ruta_img = os.path.join(carpeta_entrada, nombre_archivo)
            nombre_base = Path(nombre_archivo).stem

            try:
                img_original  = cargar_imagen(ruta_img)
                img_mejorada  = mejorar_contraste(img_original)

                # Máscara ANTES del relleno (solo para visualización comparativa)
                img_suave = gaussian(img_mejorada, sigma=SIGMA_GAUSS,
                                     preserve_range=True).astype(np.uint8)
                from skimage import filters as _f
                umbral = _f.threshold_otsu(img_suave) * (1 + UMBRAL_OFFSET)
                mascara_antes = (img_suave > umbral).astype(np.uint8) * 255

                mascara   = segmentar_hueso(img_mejorada)
                segmentada = aplicar_mascara(img_original, mascara)

                guardar_resultados(nombre_base, img_original,
                                   mascara, segmentada, carpeta_salida)
                total_procesadas += 1

                # Guardar muestra representativa (primera imagen de cada categoría)
                if len(muestras_viz) < len(CATEGORIAS) * 2:
                    muestras_viz.append({
                        "nombre":        f"{categoria}\n{nombre_archivo}",
                        "original":      img_original,
                        "mascara_antes": mascara_antes,
                        "mascara":       mascara,
                        "segmentada":    segmentada,
                    })

            except Exception as e:
                print(f"\n  ✗ Error en {nombre_archivo}: {e}")

    print("\n" + "=" * 60)
    print(f"  ✅ Total procesadas: {total_procesadas} imágenes")
    print(f"  📂 Resultados guardados en: {os.path.abspath(OUTPUT_ROOT)}")
    print("=" * 60)

    # Mostrar hasta 4 muestras (2 por clase si hay suficientes)
    visualizar_muestra(muestras_viz[:4])

    return os.path.abspath(OUTPUT_ROOT)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ruta_salida = procesar_dataset()
    print(f"\n📌 Ruta de salida: {ruta_salida}")
