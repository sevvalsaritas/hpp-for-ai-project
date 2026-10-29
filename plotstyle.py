"""
plotstyle.py -- Tüm figürlerin ortak görsel stili.

Neden? Rapordaki 5-7 figür tek bir sistem gibi okunmalı: aynı yöntem her
figürde aynı renkte (ör. NumPy/BLAS hep mavi), ince çizgiler, silik ızgara.
Renkler renk körlüğü için doğrulanmış sabit bir sıradan gelir; sırayla atanır,
asla rastgele değil.
"""

import matplotlib
matplotlib.use("Agg")            # ekransız çizim: scriptler pencere açmadan PNG üretir
import matplotlib.pyplot as plt

# Kategorik palet (sabit sıra -- renk körlüğü dostu komşu çiftler)
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Yöntem -> renk eşlemesi: bir yöntem TÜM figürlerde aynı renkte görünür.
METHOD_COLORS = {
    "python": C["red"], "python_opt": C["magenta"],
    "numpy": C["blue"], "blas": C["blue"],
    "numba": C["orange"], "numba_parallel": C["aqua"],
    "thread": C["violet"], "process": C["yellow"], "dask": C["green"], "ray": C["magenta"],
    "async": C["green"],
}


def apply():
    """Matplotlib varsayılanlarını proje stiline çevirir."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": matplotlib.cycler(color=list(C.values())),
    })


def save(fig, path):
    """Figürü 150 dpi PNG olarak kaydeder ve kapatır (bellek sızıntısını önler)."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Kaydedildi: {path}")


apply()
