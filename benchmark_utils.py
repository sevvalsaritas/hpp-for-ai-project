"""
Benchmark & Plotting Utilities for Project 1 (hocanın starter dosyası, genişletildi)

Tüm aşamalarda ortak kullanılır:
- Güvenilir zamanlama (warmup + çoklu ölçüm + MEDYAN)  <- bizim eklememiz
- Bellek ölçümü (tracemalloc)
- Doğruluk kontrolü (tolerans + eşitlik/tie farkındalığı) <- bizim eklememiz
- Grafik ve CSV/JSON dışa aktarma
"""

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import csv


# ========================================================================
# TIMING & MEMORY
# ========================================================================

def benchmark(func, *args, warmup=3, runs=5, auto_runs=False, **kwargs):
    """
    Bir fonksiyonu ısınma (warmup) + çoklu ölçümle zamanlar.

    Neden warmup? İlk çağrılar adil değildir: Numba ilk çağrıda derleme yapar,
    CPU önbellekleri soğuktur, BLAS thread havuzu henüz kurulmamıştır.
    Neden medyan? Windows'ta arka plan işlemleri tekil ölçümlerde sıçramalara
    yol açar; medyan bu aykırı değerlere ortalamadan çok daha dayanıklıdır.
    Proje kuralı: warmup + en az 5 ölçüm, medyan raporla.

    auto_runs=True: tek çağrı 50 ms'den kısaysa ölçüm sayısı 25'e çıkarılır.
    Neden? ~1 ms süren işlerde Windows'un thread uyandırma gecikmesi (zamanlayıcı
    çözünürlüğü 1-15.6 ms) tekil ölçümleri 10 kata kadar şişirebiliyor (Phase 2C'de
    gözlendi: aynı çağrı 1 ms ile 19 ms arasında). Ucuz işlerde daha çok ölçüm
    almak maliyetsizdir ve medyanı kararlı hale getirir.

    Returns:
        dict: median, mean, std, min, max, raw_times ve son çağrının sonucu
    """
    # Isınma çağrıları: sonuçları atılır, sadece JIT/cache/thread havuzu hazırlanır.
    t_last = 0.0
    for _ in range(warmup):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t_last = time.perf_counter() - t0
    if auto_runs and warmup and t_last < 0.05:
        runs = max(runs, 25)

    # Zamanlanan çağrılar: perf_counter monoton ve en yüksek çözünürlüklü saattir.
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "median": float(np.median(times)),   # raporda kullanılan ana metrik
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "warmup": warmup,
        "runs": runs,
        "results": result,
        "raw_times": times.tolist(),
    }


def strip_results(res):
    """benchmark() çıktısından büyük 'results' alanını atar (JSON'a yazmak için)."""
    return {k: v for k, v in res.items() if k != "results"}


def benchmark_with_memory(func, *args, **kwargs):
    """
    Benchmark a single run with peak memory measurement.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time": t1 - t0,
        "peak_memory_mb": peak / (1024 * 1024),
        "results": result,
    }


# ========================================================================
# CORRECTNESS
# ========================================================================

def verify_results(result_a, result_b, tol=1e-5):
    """
    İki top-k sonuç listesinin aynı olup olmadığını kontrol eder.
    Hem doküman indeksleri hem skorlar (tolerans içinde) karşılaştırılır.
    """
    if len(result_a) != len(result_b):
        return False
    for ra, rb in zip(result_a, result_b):
        if len(ra) != len(rb):
            return False
        for (ia, sa), (ib, sb) in zip(ra, rb):
            if ia != ib or abs(sa - sb) > tol:
                return False
    return True


def verify_results_tie_aware(result_a, result_b, tol=1e-5):
    """
    verify_results'ın eşitlik (tie) durumunu doğru ele alan sürümü.

    Neden gerekli? float32 ile float64 farklı yuvarlama yapar. İki dokümanın
    skoru 1e-7 kadar farklıysa sıraları uygulamalar arasında yer değiştirebilir;
    bu bir hata değildir. Burada önce skorların tolerans içinde eşit olduğunu,
    sonra indeks farklarının yalnızca skorları eşit (tol içinde) olan
    dokümanlarda olduğunu doğruluyoruz.
    Dönüş: (geçti_mi, maksimum_skor_farkı, indeksi_farklı_sorgu_sayısı)
    """
    max_diff = 0.0
    idx_mismatch = 0
    ok = len(result_a) == len(result_b)
    for ra, rb in zip(result_a, result_b):
        sa = np.array([s for _, s in ra])
        sb = np.array([s for _, s in rb])
        d = float(np.max(np.abs(sa - sb))) if len(sa) else 0.0
        max_diff = max(max_diff, d)
        if d > tol:
            ok = False
        if [i for i, _ in ra] != [i for i, _ in rb]:
            idx_mismatch += 1
            # Farklı indekslerin skorları eşit mi? Değilse gerçek bir hata vardır.
            for (ia, s1), (ib, s2) in zip(ra, rb):
                if ia != ib and abs(s1 - s2) > tol:
                    ok = False
    return ok, max_diff, idx_mismatch


def verify_top_k_indices(result_a, result_b):
    """
    Verify two results return the same top-k indices (order-independent).
    """
    if len(result_a) != len(result_b):
        return False
    for ra, rb in zip(result_a, result_b):
        if len(ra) != len(rb):
            return False
        idx_a = set(x[0] for x in ra)
        idx_b = set(x[0] for x in rb)
        if idx_a != idx_b:
            return False
    return True


# ========================================================================
# PLOTTING (Publication-Quality)
# ========================================================================

def plot_bar_comparison(names, values, ylabel, title, save_path, errors=None):
    """
    Generic bar plot with error bars.
    Use for: time comparison, speedup, memory usage.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))

    if errors:
        bars = ax.bar(x, values, yerr=errors, capsize=5,
                     color='steelblue', alpha=0.8, edgecolor='black')
    else:
        bars = ax.bar(x, values, color='steelblue', alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_speedup(names, times, baseline_name, title="Speedup vs Baseline", save_path="speedup.png"):
    """
    Plot speedup relative to a baseline.
    """
    baseline_time = dict(zip(names, times))[baseline_name]
    speedups = [baseline_time / t for t in times]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, speedups, color='forestgreen', alpha=0.8, edgecolor='black')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_scaling(x_values, y_values_dict, xlabel, ylabel, title, save_path="scaling.png"):
    """
    Line plot with multiple series.
    Use for: scaling with corpus size, workers, or components.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, y_values in y_values_dict.items():
        ax.plot(x_values, y_values, 'o-', label=label, linewidth=2, markersize=6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_amdahl(worker_counts, speedups, title="Does Amdahl's Law Fit?", save_path="amdahl.png"):
    """
    Plot measured speedup vs. Amdahl's Law prediction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Measured data
    ax.plot(worker_counts, speedups, 'o-', label='Measured', linewidth=2, markersize=8)

    # Amdahl curves
    w_range = np.linspace(1, max(worker_counts), 100)
    for P in [0.5, 0.75, 0.9, 0.99]:
        amdahl = 1 / ((1 - P) + P / w_range)
        ax.plot(w_range, amdahl, '--', alpha=0.4, label=f"Amdahl P={P}")

    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


# ========================================================================
# TABLES & DATA EXPORT
# ========================================================================

def print_comparison_table(results_dict, baseline_name="baseline"):
    """
    Print a formatted ASCII table.
    results_dict: {version_name: benchmark_result}
    """
    # Medyan varsa onu kullan (proje kuralı), yoksa eski davranış (mean/time).
    def _t(r):
        return r.get("median", r.get("mean", r.get("time", 0)))

    baseline_time = _t(results_dict.get(baseline_name, {"time": 1.0}))

    print("\n" + "=" * 80)
    print(f"{'Version':<25} {'Median (s)':<12} {'Speedup':<10} {'Std (s)':<12} {'Memory (MB)':<12}")
    print("=" * 80)
    for name, res in results_dict.items():
        mean = _t(res)
        std = res.get("std", 0)
        mem = res.get("peak_memory_mb", "--")
        speedup = baseline_time / mean if mean > 0 else 0
        mem_str = f"{mem:.1f}" if isinstance(mem, (int, float)) else str(mem)
        print(f"{name:<25} {mean:<12.4f} {speedup:<10.2f}x {std:<12.4f} {mem_str:<12}")
    print("=" * 80)


def save_results_csv(results_dict, filepath="benchmark_results.csv"):
    """Benchmark sonuçlarını rapora eklemek için CSV'ye yazar (medyan dahil)."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['version', 'median_time', 'mean_time', 'std_time',
                         'min_time', 'max_time', 'peak_memory_mb', 'raw_times'])
        for name, res in results_dict.items():
            writer.writerow([
                name,
                res.get('median', ''),
                res.get('mean', res.get('time', '')),
                res.get('std', ''),
                res.get('min', ''),
                res.get('max', ''),
                res.get('peak_memory_mb', ''),
                ";".join(f"{t:.6g}" for t in res.get('raw_times', [])),
            ])
    print(f"Saved: {filepath}")


def save_rows_csv(rows, filepath):
    """
    Liste-of-dict biçimindeki ham sonuçları CSV'ye yazar.
    Her satır bir ölçüm konfigürasyonudur (ör. yöntem x boyut x worker).
    """
    if not rows:
        return
    keys = []
    for r in rows:                       # tüm satırlardaki anahtarların birleşimi
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (";".join(f"{t:.6g}" for t in v) if isinstance(v, list) else v)
                             for k, v in r.items()})
    print(f"Saved: {filepath}")


# ========================================================================
# PROFILING HELPERS
# ========================================================================

def profile_with_cprofile(func, *args, **kwargs):
    """Run cProfile and print top 20 functions by cumulative time."""
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(20)
    print(s.getvalue())
    return result
