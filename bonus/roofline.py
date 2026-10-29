"""
Bonus: Roofline modeli -- arama çekirdeklerimiz donanımın hangi sınırına takılıyor?

Roofline: Bir çekirdeğin ulaşabileceği performans
    P = min(tepe_FLOP/s,  bant_genişliği x aritmetik_yoğunluk)
Aritmetik yoğunluk (AI) = FLOP / bellekten okunan-yazılan bayt.
  * AI düşükse (sol taraf) çekirdek BELLEK BANT GENİŞLİĞİ sınırlıdır: daha çok
    çekirdek/thread eklemek pek işe yaramaz, veri yetişmez.
  * AI yüksekse (sağ taraf) HESAP sınırlıdır: SIMD/çok çekirdek işe yarar.

Arama S = Q @ C.T (q sorgu, N doküman, d boyut, float32):
    FLOP = 2 q N d        bayt ~ 4 (N d + q d + q N)
  q = 1   -> AI ~ 0.5 FLOP/B  (her doküman bir kez kullanılır: bant genişliği sınırlı)
  q = 100 -> AI ~ 40 FLOP/B  (her doküman 100 sorguda tekrar kullanılır: hesap sınırlı)
Bu, "sorguları toplu işle" kararının (batching) roofline açıklamasıdır.

Ölçülenler: tepe FLOP/s (büyük sgemm), bant genişliği (Numba paralel toplama,
1 GB dizi), ve çekirdeklerimizin gerçek GFLOP/s değerleri.
"""

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3_parallelism"))
from benchmark_utils import benchmark  # noqa: E402
from common import (FIGURES_DIR, TOP_K, configure_threads, load_embeddings, load_json,  # noqa: E402
                    normalize_rows, save_json)
import plotstyle  # noqa: E402
from plotstyle import C, METHOD_COLORS, plt  # noqa: E402
from search_kernels import search_numba_fused  # noqa: E402


@njit(parallel=True, fastmath=True, cache=True)
def _par_sum(a):
    """Bant genişliği ölçümü: dizinin tamamını bir kez okur, neredeyse hiç hesap yapmaz."""
    s = 0.0
    for i in prange(a.shape[0]):
        s += a[i]
    return s


def measure_peaks(threads):
    """Tepe FLOP/s (4096^3 sgemm) ve okuma bant genişliği (1 GB) ölçümü."""
    n = 4096
    A = np.random.default_rng(0).standard_normal((n, n)).astype(np.float32)
    b = benchmark(lambda: A @ A, warmup=2, runs=5)
    peak = 2 * n**3 / b["median"] / 1e9
    arr = np.ones(256 * 1024**2, dtype=np.float32)          # 1 GB: önbelleğe sığmaz
    bb = benchmark(_par_sum, arr, warmup=2, runs=5)
    bw = arr.nbytes / bb["median"] / 1e9
    print(f"Tepe (sgemm {n}^3, {threads} thread): {peak:.1f} GFLOP/s | bant genişliği: {bw:.1f} GB/s")
    return peak, bw


def ai_search(q, n, d):
    """Arama çekirdeğinin aritmetik yoğunluğu (FLOP/bayt), her matris bir kez okunur/yazılır varsayımıyla."""
    return (2 * q * n * d) / (4 * (n * d + q * d + q * n))


def main():
    threads = configure_threads()
    peak, bw = measure_peaks(threads)
    C_all, Q = load_embeddings()
    Cn = np.ascontiguousarray(normalize_rows(C_all).astype(np.float32))
    Qn = np.ascontiguousarray(normalize_rows(Q).astype(np.float32))
    n, d = Cn.shape
    Qb = np.ascontiguousarray(Cn[:2000])

    pts = []
    for q, Qx, label in [(1, Qn[:1], "BLAS q=1 (tek sorgu)"), (100, Qn, "BLAS q=100"),
                         (2000, Qb, "BLAS q=2000")]:
        b = benchmark(lambda: Qx @ Cn.T, warmup=3, runs=25)
        pts.append({"label": label, "ai": ai_search(q, n, d), "gflops": 2 * q * n * d / b["median"] / 1e9,
                    "key": "blas"})
    b = benchmark(search_numba_fused, Qn, Cn, TOP_K, warmup=3, runs=25)
    pts.append({"label": "Numba fused q=100", "ai": ai_search(100, n, d),
                "gflops": 2 * 100 * n * d / b["median"] / 1e9, "key": "numba_parallel"})
    try:
        p1 = load_json("phase1_baseline.json")
        f = 2 * p1["n_queries"] * p1["n_docs"] * p1["dim"]
        pts.append({"label": "Naive saf Python", "ai": ai_search(p1["n_queries"], p1["n_docs"], p1["dim"]),
                    "gflops": f / p1["naive"]["median"] / 1e9, "key": "python"})
    except FileNotFoundError:
        pass
    for p in pts:
        roof = min(peak, bw * p["ai"])
        p["roof_gflops"] = roof
        p["pct_of_roof"] = 100 * p["gflops"] / roof
        print(f"  {p['label']:22s} AI={p['ai']:7.2f} FLOP/B  {p['gflops']:8.2f} GFLOP/s  "
              f"(tavanın %{p['pct_of_roof']:.0f}'i)")
    save_json({"threads": threads, "peak_gflops": peak, "bandwidth_gbs": bw,
               "ridge_point": peak / bw, "points": pts}, "bonus_roofline.json")
    plot(peak, bw, pts, FIGURES_DIR / "figB_roofline.png")


def plot(peak, bw, pts, save_path):
    """Roofline grafiği: eğik çizgi = bant genişliği tavanı, yatay = hesap tavanı."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ai = np.logspace(-2, 3, 200)
    ax.plot(ai, np.minimum(peak, bw * ai), color=plotstyle.INK2, linewidth=2)
    ax.text(0.02, bw * 0.02 * 1.4, f"bant genişliği {bw:.0f} GB/s", rotation=29, fontsize=8.5,
            color=plotstyle.INK2)
    ax.text(200, peak * 1.12, f"tepe {peak:.0f} GFLOP/s", fontsize=8.5, color=plotstyle.INK2, ha="center")
    for p in pts:
        col = METHOD_COLORS.get(p["key"], C["violet"])
        ax.scatter(p["ai"], p["gflops"], s=70, color=col, zorder=5, edgecolor="#fcfcfb", linewidth=1.5)
        ax.annotate(p["label"], (p["ai"], p["gflops"]), textcoords="offset points", xytext=(7, -4),
                    fontsize=8.3, color=plotstyle.INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aritmetik yoğunluk (FLOP / bayt, log)")
    ax.set_ylabel("Performans (GFLOP/s, log)")
    ax.set_title("Roofline: tek sorgu bant genişliği sınırlı, toplu sorgu hesap sınırlı")
    ax.set_ylim(bottom=min(p["gflops"] for p in pts) / 3)
    plt.tight_layout()
    plotstyle.save(fig, save_path)


if __name__ == "__main__":
    main()
