"""
Phase 4: Final sistem -- ölçekleme testi, doğruluk kontrolü, tüm sürümlerin karşılaştırması.

1. Ölçekleme: korpus boyutu 1K .. N_toplam (common.scaling_sizes) için
   - naive baseline (saf Python, 100 sorgu; sorgu başına süre raporlanır)
   - final motor: kesin (384 boyut) ve SVD-256 modu, 100 sorgu toplu + tek sorgu gecikmesi
   -> Figür 6
2. Doğruluk: final motorun top-5'i naive baseline ile (tolerans 1e-5)
   - kesin mod: 384 boyutlu orijinal embedding'lerde
   - SVD modu: baseline aynı indirgenmiş vektörlerde çalıştırılır (aynı uzay)
3. Tüm sürümler standart iş yükünde (100 sorgu x tüm korpus) -> Figür 7
"""

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, ROOT / "phase1_baseline", ROOT / "phase3_parallelism"):
    sys.path.insert(0, str(p))

import numba  # noqa: E402

from benchmark_utils import benchmark, save_rows_csv, verify_results_tie_aware  # noqa: E402
from common import (FIGURES_DIR, REF_N_QUERIES, RESULTS_DIR, TOP_K, configure_threads,  # noqa: E402
                    load_embeddings, load_json, normalize_rows, save_json, scaling_sizes)
import plotstyle  # noqa: E402
from plotstyle import C, METHOD_COLORS, plt  # noqa: E402
from naive_search import search_naive  # noqa: E402
from search_engine import SimilaritySearchEngine  # noqa: E402
from search_kernels import search_block, search_numba_fused, to_result_list  # noqa: E402
from process_pool_search import SharedCorpusPool  # noqa: E402
from thread_pool_search import search_threadpool  # noqa: E402

SVD_K = 256
N_BASE_Q = REF_N_QUERIES   # baseline da final motorla aynı 100 sorguyla ölçülür


def scaling_test(C_all, Q, sizes):
    """
    Her korpus boyutunda baseline ve final motor sürelerini ölçer.

    Standart iş yükü noktasında (100 sorgu x tüm korpus) naive yeniden ölçülmez,
    Phase 1'in ölçümü kullanılır: aynı iş için raporda tek bir baseline sayısı
    olsun (saf Python süresi laptopta çalıştırmalar arası belirgin dalgalanıyor;
    iki ayrı ölçüm Figür 6 ile Figür 7'yi çelişkili gösterirdi).
    """
    rows = []
    q_list_base = Q[:N_BASE_Q].tolist()
    p1 = load_json("phase1_baseline.json")
    for n in sizes:
        r = {"n_docs": n}
        # --- naive baseline (sorgu başına saniye) ---
        if n == p1["n_docs"] and N_BASE_Q == p1["n_queries"]:
            r["naive_per_query_s"] = p1["naive"]["median"] / N_BASE_Q
            r["raw_naive"] = p1["naive"]["raw_times"]
            r["naive_source"] = "phase1_baseline.json"
        else:
            c_list = C_all[:n].tolist()
            b = benchmark(search_naive, q_list_base, c_list, TOP_K, warmup=1, runs=5)
            r["naive_per_query_s"] = b["median"] / N_BASE_Q
            r["raw_naive"] = b["raw_times"]
        # --- final motor: kesin ve SVD modu ---
        for mode, k_svd in (("exact", None), ("svd", SVD_K)):
            eng = SimilaritySearchEngine(C_all[:n], svd_k=k_svd)
            Qn = eng.prepare_queries(Q)
            bb = benchmark(eng.search_prepared, Qn, TOP_K, warmup=3, runs=5, auto_runs=True)
            b1 = benchmark(eng.search_prepared, Qn[:1], TOP_K, warmup=3, runs=25)
            r[f"{mode}_batch100_s"] = bb["median"]
            r[f"{mode}_per_query_s"] = bb["median"] / len(Qn)
            r[f"{mode}_single_query_s"] = b1["median"]
            r[f"{mode}_index_s"] = eng.index_time_s
            r[f"{mode}_mem_mb"] = eng.C.nbytes / 1024**2
            r[f"raw_{mode}"] = bb["raw_times"]
        r["speedup_exact_vs_naive"] = r["naive_per_query_s"] / r["exact_per_query_s"]
        rows.append(r)
        print(f"  N={n:6d}: naive {r['naive_per_query_s'] * 1e3:9.1f} ms/sorgu | exact "
              f"{r['exact_per_query_s'] * 1e6:7.1f} us/sorgu (tek sorgu {r['exact_single_query_s'] * 1e3:.3f} ms)"
              f" | svd {r['svd_per_query_s'] * 1e6:7.1f} us/sorgu | {r['speedup_exact_vs_naive']:,.0f}x")
    return rows


def loglog_slope(x, y):
    """log-log eğim: 1 -> doğrusal ölçekleme, 2 -> karesel."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def correctness(C_all, Q):
    """Final motor vs naive baseline (tolerans 1e-5), tüm korpus ve 100 sorgu üzerinde."""
    out = {}
    n = C_all.shape[0]
    # Kesin mod
    eng = SimilaritySearchEngine(C_all)
    fast = to_result_list(*eng.search(Q))
    t0 = time.perf_counter()
    base = search_naive(Q.tolist(), C_all.tolist(), TOP_K)
    t_base = time.perf_counter() - t0
    ok, diff, mism = verify_results_tie_aware(fast, base, tol=1e-5)
    out["exact"] = {"n_docs": n, "n_queries": len(Q), "passed": ok, "max_abs_score_diff": diff,
                    "queries_with_index_order_diff": mism, "naive_time_s": t_base}
    print(f"  kesin mod: geçti={ok}, max skor farkı={diff:.2e}, sıra farkı olan sorgu={mism}")
    # SVD modu: baseline aynı indirgenmiş uzayda
    eng_s = SimilaritySearchEngine(C_all, svd_k=SVD_K)
    fast_s = to_result_list(*eng_s.search(Q))
    Cr = eng_s.svd.transform(C_all).astype(np.float32)
    Qr = eng_s.svd.transform(Q).astype(np.float32)
    base_s = search_naive(Qr.tolist(), Cr.tolist(), TOP_K)
    ok_s, diff_s, mism_s = verify_results_tie_aware(fast_s, base_s, tol=1e-5)
    # Kalite: SVD modunun top-5'i, kesin 384 boyutlu top-5 ile ne kadar örtüşüyor?
    rec = np.mean([len({i for i, _ in a} & {i for i, _ in b}) / TOP_K for a, b in zip(fast_s, fast)])
    out["svd"] = {"k": SVD_K, "passed": ok_s, "max_abs_score_diff": diff_s,
                  "queries_with_index_order_diff": mism_s, "recall_at_5_vs_exact": float(rec)}
    print(f"  SVD-{SVD_K} modu: geçti={ok_s}, max fark={diff_s:.2e}, recall@5 (kesine göre)={rec:.3f}")
    return out


def all_versions(C_all, Q, n):
    """Standart iş yükü (100 sorgu x n = tüm korpus) üzerinde hikayedeki tüm sürümler -> Figür 7."""
    p1 = load_json("phase1_baseline.json")
    # Saf Python satırları Phase 1'den gelir; aynı iş yükünde ölçülmüş olmaları şart.
    assert p1["n_docs"] == n and p1["n_queries"] == len(Q), "Phase 1 standart iş yüküyle yeniden çalıştırılmalı"
    Cn = np.ascontiguousarray(normalize_rows(C_all[:n]).astype(np.float32))
    Qn = np.ascontiguousarray(normalize_rows(Q).astype(np.float32))
    ref = search_block(Qn, Cn, TOP_K)
    rows = [{"version": "Saf Python (naive)", "time_s": p1["naive"]["median"], "key": "python"},
            {"version": "Saf Python + algoritmik", "time_s": p1["python_optimized"]["median"],
             "key": "python_opt"}]

    def add(name, key, fn, blas=8, nb=8):
        """Thread ayarları ölçüm DIŞINDA yapılır (threadpool_limits girişi ~10 ms sürer)."""
        numba.set_num_threads(nb)
        with threadpool_limits(blas):
            b = benchmark(fn, warmup=3, runs=5, auto_runs=True)
        ok, diff, _ = verify_results_tie_aware(to_result_list(*b["results"]), to_result_list(*ref))
        rows.append({"version": name, "time_s": b["median"], "key": key, "correct": ok,
                     "raw_times": b["raw_times"]})
        print(f"  {name:36s} {b['median'] * 1e3:9.3f} ms  doğru={ok}")

    add("Numba fused (1 thread)", "numba", lambda: search_numba_fused(Qn, Cn, TOP_K), nb=1)
    add("NumPy/BLAS + argpartition (1 thread)", "numpy", lambda: search_block(Qn, Cn, TOP_K), blas=1)
    add("NumPy/BLAS + argpartition (16 thread)", "numpy", lambda: search_block(Qn, Cn, TOP_K), blas=16)
    add("NumPy/BLAS + argpartition (8 thread)", "numpy", lambda: search_block(Qn, Cn, TOP_K), blas=8)
    add("Numba fused prange (16 thread)", "numba_parallel", lambda: search_numba_fused(Qn, Cn, TOP_K), nb=16)
    add("ThreadPool 8 x BLAS-1", "thread", lambda: search_threadpool(Qn, Cn, TOP_K, 8), blas=1)
    pool = SharedCorpusPool(Cn, 8)
    try:
        add("ProcessPool 8 + shm", "process", lambda: pool.search(Qn, TOP_K))
    finally:
        pool.close()
    eng = SimilaritySearchEngine(C_all[:n])          # 8 thread BLAS + Numba top-k
    add("FİNAL: BLAS 8 thr + Numba top-k", "final", lambda: eng.search_prepared(Qn, TOP_K))
    numba.set_num_threads(configure_threads())
    return rows


def plot_scaling(rows, save_path):
    """Figür 6: sorgu başına süre vs korpus boyutu (log-log), baseline ve final."""
    n = [r["n_docs"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    series = [("naive_per_query_s", "Naive saf Python", METHOD_COLORS["python"]),
              ("exact_single_query_s", "Final, tek sorgu gecikmesi", C["violet"]),
              ("exact_per_query_s", "Final kesin (100 sorgu toplu, sorgu başına)", METHOD_COLORS["numpy"]),
              ("svd_per_query_s", f"Final SVD-{SVD_K} (toplu, sorgu başına)", C["aqua"])]
    slopes = {}
    for key, label, col in series:
        y = [r[key] * 1e3 for r in rows]
        slopes[key] = loglog_slope(n, y)
        ax.plot(n, y, "o-", color=col, label=f"{label}  (eğim {slopes[key]:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Korpus boyutu (doküman, log)")
    ax.set_ylabel("Sorgu başına süre (ms, log, medyan)")
    ax.set_title("(a) Ölçekleme: log-log eğim 1 = doğrusal")
    ax.legend(fontsize=8.2, loc="center right")

    ax = axes[1]
    sp = [r["speedup_exact_vs_naive"] for r in rows]
    bars = ax.bar([f"{x / 1000:g}K" for x in n], sp, color=METHOD_COLORS["numpy"], width=0.6)
    for b_, s in zip(bars, sp):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{s:,.0f}x", ha="center",
                va="bottom", fontsize=9, color=plotstyle.INK)
    ax.set_xlabel("Korpus boyutu")
    ax.set_ylabel("Hızlanma (naive / final kesin)")
    ax.set_title("(b) Final sistemin naive'e göre hızlanması")
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    plotstyle.save(fig, save_path)
    return slopes


def plot_all_versions(rows, save_path, n):
    """Figür 7: tüm sürümler, aynı iş yükü, log ölçekli yatay çubuk."""
    rows = sorted(rows, key=lambda r: -r["time_s"])
    t0 = rows[0]["time_s"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    cols = {**METHOD_COLORS, "final": C["green"]}
    y = np.arange(len(rows))
    ax.barh(y, [r["time_s"] * 1e3 for r in rows], color=[cols[r["key"]] for r in rows], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r["version"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    for yi, r in zip(y, rows):
        ax.text(r["time_s"] * 1e3 * 1.15, yi, f"{r['time_s'] * 1e3:,.2f} ms   ({t0 / r['time_s']:,.0f}x)",
                va="center", fontsize=8.6, color=plotstyle.INK)
    ax.set_xlim(right=t0 * 1e3 * 40)
    n_tr = f"{n:,}".replace(",", ".")                 # Türkçe binlik ayıracı (sadece sayıda)
    ax.set_xlabel(f"100 sorgu x {n_tr} doküman için süre (ms, log, medyan)")
    ax.set_title("Tüm sürümler aynı iş yükünde (parantez: naive'e göre hızlanma)")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plotstyle.save(fig, save_path)


def main():
    configure_threads()
    C_all, Q = load_embeddings()
    sizes = scaling_sizes(C_all.shape[0])
    print(f"Korpus {C_all.shape}, ölçekleme boyutları {sizes}")

    print("\n== Ölçekleme testi ==")
    rows = scaling_test(C_all, Q, sizes)
    slopes = plot_scaling(rows, FIGURES_DIR / "fig6_scaling.png")

    print("\n== Doğruluk kontrolü (tolerans 1e-5) ==")
    corr = correctness(C_all, Q)

    print(f"\n== Tüm sürümler (standart iş yükü: {len(Q)} sorgu x {C_all.shape[0]} doküman) ==")
    vers = all_versions(C_all, Q, n=C_all.shape[0])
    plot_all_versions(vers, FIGURES_DIR / "fig7_all_versions.png", n=C_all.shape[0])

    save_rows_csv(rows, RESULTS_DIR / "phase4_scaling.csv")
    save_rows_csv(vers, RESULTS_DIR / "phase4_all_versions.csv")
    save_json({"sizes": sizes, "scaling": rows, "loglog_slopes": slopes, "correctness": corr,
               "all_versions": vers, "svd_k": SVD_K}, "phase4_final.json")


if __name__ == "__main__":
    main()
