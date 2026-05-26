import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  Rutas
# ─────────────────────────────────────────────────────────────
CSV_ENTRADA    = os.path.join("/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Características", "caracteristicas.csv")
OUTPUT_VIZ     = "/home/manguito/Code/University/Reconocimiento de Patrones/DataSet Full/Resultados Características/Gráficas Características"

# Paleta de colores clínica
COLOR_BENIGNO  = "#2196F3"   # Azul
COLOR_MALIGNO  = "#F44336"   # Rojo
PALETA         = {0: COLOR_BENIGNO, 1: COLOR_MALIGNO}
NOMBRE_CLASE   = {0: "Benigno", 1: "Maligno"}

# Grupos de características para graficar por separado
GRUPOS = {
    "GLCM":          lambda c: [x for x in c if x.startswith("glcm_")],
    "Haralick":      lambda c: [x for x in c if x.startswith("har_")],
    "Estadísticas":  lambda c: [x for x in c if x.startswith("stat_")],
    "Morfología":    lambda c: [x for x in c if x.startswith("morfo_")],
}

DPI = 150
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  ESTILOS
# ══════════════════════════════════════════════════════════════

def configurar_estilo():
    plt.rcParams.update({
        "figure.facecolor":   "#0d1117",
        "axes.facecolor":     "#161b22",
        "axes.edgecolor":     "#30363d",
        "axes.labelcolor":    "#c9d1d9",
        "xtick.color":        "#8b949e",
        "ytick.color":        "#8b949e",
        "grid.color":         "#21262d",
        "grid.linewidth":     0.8,
        "text.color":         "#c9d1d9",
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     11,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
    })

LEYENDA = [
    mpatches.Patch(color=COLOR_BENIGNO, label="Benigno"),
    mpatches.Patch(color=COLOR_MALIGNO, label="Maligno"),
]


# ══════════════════════════════════════════════════════════════
#  BOXPLOTS
# ══════════════════════════════════════════════════════════════

def graficar_boxplots(df: pd.DataFrame, columnas: list,
                      titulo: str, ruta_salida: str):
    """Genera boxplots para un grupo de características."""
    n = len(columnas)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.5),
                             facecolor="#0d1117")
    fig.suptitle(f"Boxplots — {titulo}", fontsize=14, color="white",
                 fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(columnas):
        ax = axes_flat[i]
        ax.set_facecolor("#161b22")

        datos_por_clase = [
            df.loc[df["etiqueta"] == et, col].dropna().values
            for et in [0, 1]
        ]

        bp = ax.boxplot(
            datos_por_clase,
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#8b949e", linewidth=1.2),
            capprops=dict(color="#8b949e", linewidth=1.5),
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor="#8b949e", alpha=0.5),
        )

        for patch, color in zip(bp["boxes"], [COLOR_BENIGNO, COLOR_MALIGNO]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Test de Mann-Whitney
        if len(datos_por_clase[0]) > 0 and len(datos_por_clase[1]) > 0:
            _, p_val = stats.mannwhitneyu(
                datos_por_clase[0], datos_por_clase[1], alternative="two-sided"
            )
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                  ("*" if p_val < 0.05 else "ns"))
            ax.set_title(f"{col}\n(p={p_val:.3f} {sig})",
                         color="#c9d1d9", fontsize=8, pad=4)
        else:
            ax.set_title(col, color="#c9d1d9", fontsize=8, pad=4)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Benigno", "Maligno"], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[:].set_edgecolor("#30363d")

    # Ocultar axes sobrantes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.legend(handles=LEYENDA, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), fontsize=9,
               framealpha=0.3, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white")

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=DPI, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f" {os.path.basename(ruta_salida)}")


# ══════════════════════════════════════════════════════════════
#  VIOLIN PLOTS
# ══════════════════════════════════════════════════════════════

def graficar_violins(df: pd.DataFrame, columnas: list,
                     titulo: str, ruta_salida: str):
    """Genera violin plots para un grupo de características."""
    n = len(columnas)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.8),
                             facecolor="#0d1117")
    fig.suptitle(f"Violin Plots — {titulo}", fontsize=14, color="white",
                 fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    # Preparar DataFrame largo para seaborn
    df_plot = df[["etiqueta"] + columnas].copy()
    df_plot["Clase"] = df_plot["etiqueta"].map(NOMBRE_CLASE)

    for i, col in enumerate(columnas):
        ax = axes_flat[i]
        ax.set_facecolor("#161b22")

        sub = df_plot[["Clase", col]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(
            data=sub, x="Clase", y=col,
            palette={"Benigno": COLOR_BENIGNO, "Maligno": COLOR_MALIGNO},
            inner="box", cut=0, linewidth=1.2,
            order=["Benigno", "Maligno"], ax=ax,
        )

        # Superponer puntos individuales (jitter)
        for j, clase in enumerate(["Benigno", "Maligno"]):
            valores = sub.loc[sub["Clase"] == clase, col].values
            jitter  = np.random.uniform(-0.12, 0.12, len(valores))
            color   = COLOR_BENIGNO if clase == "Benigno" else COLOR_MALIGNO
            ax.scatter(j + jitter, valores, alpha=0.35, s=10,
                       color=color, zorder=3)

        # p-valor
        g0 = sub.loc[sub["Clase"] == "Benigno",  col].values
        g1 = sub.loc[sub["Clase"] == "Maligno",  col].values
        if len(g0) > 1 and len(g1) > 1:
            _, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                  ("*" if p_val < 0.05 else "ns"))
            ax.set_title(f"{col}\n(p={p_val:.3f} {sig})",
                         color="#c9d1d9", fontsize=8, pad=4)
        else:
            ax.set_title(col, color="#c9d1d9", fontsize=8, pad=4)

        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[:].set_edgecolor("#30363d")
        ax.tick_params(colors="#8b949e")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=DPI, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"{os.path.basename(ruta_salida)}")


# ══════════════════════════════════════════════════════════════
#  MATRIZ DE CORRELACIÓN
# ══════════════════════════════════════════════════════════════

def graficar_correlacion(df: pd.DataFrame, columnas: list,
                         titulo: str, ruta_salida: str,
                         max_feats: int = None):
    """Genera la matriz de correlación de Pearson."""
    cols = columnas if max_feats is None else columnas[:max_feats]
    if len(cols) < 2:
        return

    matriz = df[cols].corr(method="pearson")

    fig_size = max(10, len(cols) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85),
                            facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    mascara = np.triu(np.ones_like(matriz, dtype=bool))

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        matriz,
        mask=mascara,
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=(len(cols) <= 20),
        fmt=".2f",
        linewidths=0.3,
        linecolor="#21262d",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Correlación de Pearson"},
        annot_kws={"size": 7},
    )

    ax.set_title(f"Matriz de Correlación — {titulo}",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#8b949e", labelsize=7, rotation=45)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Estilo de la colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#8b949e", labelsize=8)
    cbar.set_label("Correlación de Pearson", color="#c9d1d9", fontsize=9)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=DPI, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"  💾 {os.path.basename(ruta_salida)}")


# ══════════════════════════════════════════════════════════════
#  DISCRIMINANTE DE CARACTERÍSTICAS
# ══════════════════════════════════════════════════════════════

def graficar_ranking_discriminante(df: pd.DataFrame, columnas: list,
                                   ruta_salida: str, top_n: int = 20):
                                       
    resultados = []
    g0 = df.loc[df["etiqueta"] == 0]
    g1 = df.loc[df["etiqueta"] == 1]

    for col in columnas:
        v0 = g0[col].dropna().values
        v1 = g1[col].dropna().values
        if len(v0) < 2 or len(v1) < 2:
            continue
        try:
            _, p_val = stats.mannwhitneyu(v0, v1, alternative="two-sided")
            # Cohen's d
            pooled_std = np.sqrt(
                ((len(v0) - 1) * np.std(v0, ddof=1)**2 +
                 (len(v1) - 1) * np.std(v1, ddof=1)**2) /
                (len(v0) + len(v1) - 2)
            )
            cohen_d = abs(np.mean(v0) - np.mean(v1)) / (pooled_std + 1e-10)
            resultados.append({
                "caracteristica": col,
                "p_valor":        p_val,
                "cohen_d":        cohen_d,
                "-log10_p":       -np.log10(p_val + 1e-300),
            })
        except Exception:
            continue

    if not resultados:
        print("  ⚠ No se pudo calcular el ranking discriminante.")
        return

    df_rank = pd.DataFrame(resultados).sort_values("cohen_d", ascending=False)
    df_top  = df_rank.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.42)),
                              facecolor="#0d1117")

    # — Panel izquierdo: Cohen's d
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    colores_d = [COLOR_MALIGNO if d > 0.8 else
                 ("#FF9800" if d > 0.5 else COLOR_BENIGNO)
                 for d in df_top["cohen_d"]]
    bars1 = ax1.barh(range(len(df_top)), df_top["cohen_d"].values,
                     color=colores_d, alpha=0.8, height=0.7)
    ax1.set_yticks(range(len(df_top)))
    ax1.set_yticklabels(df_top["caracteristica"].values, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Cohen's d", color="#c9d1d9")
    ax1.set_title("Tamaño del Efecto (Cohen's d)", color="white",
                  fontsize=11, fontweight="bold")
    ax1.axvline(x=0.5, color="#FF9800", linestyle="--", alpha=0.6,
                linewidth=1.2, label="Efecto medio (0.5)")
    ax1.axvline(x=0.8, color=COLOR_MALIGNO, linestyle="--", alpha=0.6,
                linewidth=1.2, label="Efecto grande (0.8)")
    ax1.legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="white")
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[:].set_edgecolor("#30363d")

    for bar, val in zip(bars1, df_top["cohen_d"].values):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=7, color="#c9d1d9")

    # — Panel derecho: -log10(p-valor)
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    colores_p = [COLOR_MALIGNO if p > -np.log10(0.001) else
                 ("#FF9800" if p > -np.log10(0.05) else COLOR_BENIGNO)
                 for p in df_top["-log10_p"]]
    bars2 = ax2.barh(range(len(df_top)), df_top["-log10_p"].values,
                     color=colores_p, alpha=0.8, height=0.7)
    ax2.set_yticks(range(len(df_top)))
    ax2.set_yticklabels(df_top["caracteristica"].values, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("-log₁₀(p-valor)", color="#c9d1d9")
    ax2.set_title("Significancia Estadística (-log₁₀ p)", color="white",
                  fontsize=11, fontweight="bold")
    ax2.axvline(x=-np.log10(0.05),  color="#FF9800",    linestyle="--",
                alpha=0.6, linewidth=1.2, label="p=0.05")
    ax2.axvline(x=-np.log10(0.001), color=COLOR_MALIGNO, linestyle="--",
                alpha=0.6, linewidth=1.2, label="p=0.001")
    ax2.legend(fontsize=8, framealpha=0.3, facecolor="#161b22",
               edgecolor="#30363d", labelcolor="white")
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines[:].set_edgecolor("#30363d")

    fig.suptitle(f"Ranking de Características Discriminantes (Top {top_n})",
                 fontsize=14, color="white", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=DPI, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f" {os.path.basename(ruta_salida)}")

    # Imprimir tabla resumen
    print("\n Top características más discriminantes:")
    print(df_top[["caracteristica", "cohen_d", "p_valor"]].to_string(index=False))

    return df_rank


# ══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

def visualizar():
    print("=" * 60)
    print("  VISUALIZACIÓN — SARCOMAS ÓSEOS")
    print("=" * 60)

    if not os.path.isfile(CSV_ENTRADA):
        print(f"\n✗ CSV no encontrado: {CSV_ENTRADA}")
        print("  Ejecuta primero el Script 2.")
        return

    df = pd.read_csv(CSV_ENTRADA)
    print(f"\n Dataset cargado: {len(df)} muestras, {df.shape[1]} columnas")
    print(f"   Benignos:  {(df['etiqueta']==0).sum()}  |  Malignos: {(df['etiqueta']==1).sum()}")

    os.makedirs(OUTPUT_VIZ, exist_ok=True)
    configurar_estilo()

    # Separar columnas de características
    cols_meta = ["imagen", "categoria", "etiqueta"]
    cols_feat = [c for c in df.columns if c not in cols_meta]

    # Obtener grupos
    grupos_cols = {
        nombre: fn(cols_feat)
        for nombre, fn in GRUPOS.items()
    }

    print("\n Grupos de características:")
    for g, cols in grupos_cols.items():
        print(f"   {g}: {len(cols)} características")

    idx = 1

    # ── Boxplots
    print("\n Generando Boxplots...")
    for nombre, cols in grupos_cols.items():
        if cols:
            ruta = os.path.join(OUTPUT_VIZ, f"{idx:02d}_boxplots_{nombre.lower()}.png")
            graficar_boxplots(df, cols, nombre, ruta)
            idx += 1

    # ── Violin plots
    print("\n Generando Violin Plots...")
    for nombre, cols in grupos_cols.items():
        if cols:
            ruta = os.path.join(OUTPUT_VIZ, f"{idx:02d}_violins_{nombre.lower()}.png")
            graficar_violins(df, cols, nombre, ruta)
            idx += 1

    # ── Matriz de correlación completa
    print("\n Generando Matrices de Correlación...")
    ruta_corr_full = os.path.join(OUTPUT_VIZ, f"{idx:02d}_correlacion_completa.png")
    graficar_correlacion(df, cols_feat, "Todas las características",
                         ruta_corr_full, max_feats=40)
    idx += 1

    # ── Correlación top 20 más discriminantes
    df_rank = graficar_ranking_discriminante(
        df, cols_feat,
        ruta_salida=os.path.join(OUTPUT_VIZ, f"{idx:02d}_ranking_discriminante.png"),
        top_n=20
    )

    if df_rank is not None:
        top20 = df_rank.head(20)["caracteristica"].tolist()
        ruta_corr_top = os.path.join(OUTPUT_VIZ, f"{idx+1:02d}_correlacion_top20.png")
        graficar_correlacion(df, top20, "Top 20 características discriminantes",
                             ruta_corr_top)

    print("\n" + "=" * 60)
    print(f" Visualizaciones guardadas en: {os.path.abspath(OUTPUT_VIZ)}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    visualizar()
