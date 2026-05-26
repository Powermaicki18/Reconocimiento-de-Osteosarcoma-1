import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage import exposure, morphology, filters, measure
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from tqdm import tqdm
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Rutas a las carpetas
# ─────────────────────────────────────────────────────────────
DATASET_ROOT   = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/dataset 1" # Carpeta raíz del dataset
OUTPUT_ROOT    = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Segmentación" # Carpeta de salida
CATEGORIAS     = ["Sarcomas Benignoss", "Sarcomas Malignos"]
EXTENSIONES    = (".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp")

# "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/dataset 1/Sarcomas Benignos"

# Parámetros de segmentación
CLAHE_CLIP      = 3.0    # Límite de contraste para CLAHE
CLAHE_GRID      = (8, 8) # Tamaño de la cuadrícula CLAHE
UMBRAL_OFFSET   = 0.05   # Offset sobre el umbral de Otsu (-1 a 1)
MIN_AREA_RATIO  = 0.01   # Área mínima de región como fracción de la imagen
MAX_AREAS       = 3      # Máximo de regiones a conservar (la más grande)
SIGMA_GAUSS     = 2.0    # Suavizado gaussiano previo a la segmentación
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


def segmentar_hueso(img: np.ndarray) -> np.ndarray:
    # 1. Suavizado
    img_suave = gaussian(img, sigma=SIGMA_GAUSS, preserve_range=True).astype(np.uint8)

    # 2. Umbralización de Otsu
    umbral_otsu = filters.threshold_otsu(img_suave)
    umbral_final = umbral_otsu * (1 + UMBRAL_OFFSET)
    mascara = (img_suave > umbral_final).astype(np.uint8) * 255

    # 3. Operaciones morfológicas
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_cierre, iterations=3)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel_apertura, iterations=2)

    # 4. Rellenar huecos internos
    mascara_inv = cv2.bitwise_not(mascara)
    h, w = mascara.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mascara_inv, flood_mask, (0, 0), 255)
    mascara_inv_flood = cv2.bitwise_not(mascara_inv)
    mascara = cv2.bitwise_or(mascara, mascara_inv_flood)

    # 5. Conservar únicamente las regiones más grandes
    etiquetas = measure.label(mascara > 0)
    propiedades = measure.regionprops(etiquetas)
    area_imagen = img.shape[0] * img.shape[1]
    area_min = area_imagen * MIN_AREA_RATIO

    # Ordenar por área descendente y tomar las MAX_AREAS más grandes
    regiones_validas = sorted(
        [r for r in propiedades if r.area >= area_min],
        key=lambda x: x.area, reverse=True
    )[:MAX_AREAS]

    mascara_final = np.zeros_like(mascara)
    for region in regiones_validas:
        mascara_final[etiquetas == region.label] = 255

    return mascara_final


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
    """Muestra una grilla con ejemplos: original | máscara | segmentada."""
    n = len(muestras)
    if n == 0:
        print("No hay muestras para visualizar.")
        return

    fig = plt.figure(figsize=(15, 5 * n), facecolor="#0d1117")
    fig.suptitle(titulo, fontsize=16, color="white", y=1.01)

    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.4, wspace=0.15)
    columnas = ["Original", "Máscara", "Segmentada"]

    for i, muestra in enumerate(muestras):
        for j, (clave, col_titulo) in enumerate(zip(
                ["original", "mascara", "segmentada"], columnas)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(muestra[clave], cmap="gray")
            ax.set_title(f"{col_titulo}\n{muestra['nombre']}", color="white",
                         fontsize=9, pad=4)
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
            print(f"\n Carpeta no encontrada: {carpeta_entrada} — se omite.")
            continue

        # Listar imágenes válidas
        imagenes = [
            f for f in os.listdir(carpeta_entrada)
            if f.lower().endswith(EXTENSIONES)
        ]

        if not imagenes:
            print(f"\n No se encontraron imágenes en: {carpeta_entrada}")
            continue

        print(f"\n Procesando: {categoria}  ({len(imagenes)} imágenes)")
        os.makedirs(carpeta_salida, exist_ok=True)

        for nombre_archivo in tqdm(imagenes, desc=f"  {categoria[:25]}", unit="img"):
            ruta_img = os.path.join(carpeta_entrada, nombre_archivo)
            nombre_base = Path(nombre_archivo).stem

            try:
                img_original = cargar_imagen(ruta_img)
                img_mejorada = mejorar_contraste(img_original)
                mascara      = segmentar_hueso(img_mejorada)
                segmentada   = aplicar_mascara(img_original, mascara)

                guardar_resultados(nombre_base, img_original,
                                   mascara, segmentada, carpeta_salida)
                total_procesadas += 1

                # Guardar muestra representativa (primera imagen de cada categoría)
                if len(muestras_viz) < len(CATEGORIAS) * 2:
                    muestras_viz.append({
                        "nombre":     f"{categoria}\n{nombre_archivo}",
                        "original":   img_original,
                        "mascara":    mascara,
                        "segmentada": segmentada,
                    })

            except Exception as e:
                print(f"\n  ✗ Error en {nombre_archivo}: {e}")

    print("\n" + "=" * 60)
    print(f" Total procesadas: {total_procesadas} imágenes")
    print(f" Resultados guardados en: {os.path.abspath(OUTPUT_ROOT)}")
    print("=" * 60)

    # Mostrar hasta 4 muestras (2 por clase si hay suficientes)
    visualizar_muestra(muestras_viz[:4])

    return os.path.abspath(OUTPUT_ROOT)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ruta_salida = procesar_dataset()
    print(f"\n Ruta de salida: {ruta_salida}")
