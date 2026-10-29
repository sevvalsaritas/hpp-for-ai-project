"""
Phase 2B: Boyut İndirgeme -- Seç ve Gerekçelendir (starter dosyası, tamamlandı)

Matris: MiniLM embedding matrisi X (N x 384). Arama maliyeti O(Q * N * d)
olduğundan d'yi 384'ten k'ya indirmek aramayı doğrudan d/k kat ucuzlatır.

Karşılaştırılan yöntemler:
  1. power_iteration_svd  -- KENDİ uygulamamız (blok güç iterasyonu + QR)
  2. np.linalg.svd        -- tam SVD (LAPACK), doğruluk referansı (ground truth)
  3. sklearn TruncatedSVD -- randomize çözücü, sadece ilk k bileşen
  4. sklearn PCA          -- veriyi ortalar (merkezler), sonra SVD

Her yöntem için: süre-k, süre-matris boyutu, k=50/100/200'de Frobenius
yeniden oluşturma hatası, tepe bellek. Ek olarak: indirgenmiş uzayda arama
yapınca top-5 sonuçların ne kadarı korunuyor (recall@5)? -> k seçimini bu belirler.

Kavramlar: BLAS-3 (matris-matris) vs BLAS-2, bellek, algoritmik karmaşıklık.
"""

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import benchmark, save_rows_csv  # noqa: E402
from common import (FIGURES_DIR, RESULTS_DIR, SEED, TOP_K, configure_threads,  # noqa: E402
                    load_embeddings, normalize_rows, save_json, scaling_sizes)
import plotstyle  # noqa: E402
from plotstyle import C, plt  # noqa: E402


def power_iteration_svd(X, k, max_iter=200, tol=1e-6, oversample=10, ritz=True,
                        seed=SEED, return_iters=False):
    """
    İlk k tekil vektörü BLOK GÜÇ İTERASYONU (orthogonal/subspace iteration) ile bulur.

    Fikir: G = X^T X matrisini bir vektöre tekrar tekrar uygularsak vektör en
    büyük özdeğerin (= sigma_1^2) yönüne döner. k vektörü aynı anda iteratif
    olarak çarpıp her adımda QR ile birbirine dik tutarsak ilk k yönü buluruz.

    Neden önce G = X^T X (d x d) hesaplıyoruz?
      X "uzun-ince" (N=19.671 >> d=384). G'yi bir kez hesaplamak O(N d^2), sonra
      her iterasyon sadece O(d^2 k). Her iterasyonda X^T (X V) yapsaydık
      iterasyon başına O(N d k) öderdik -> 100 iterasyonda çok daha pahalı.
      Ayrıca X^T X bir BLAS-3 (GEMM) çağrısı: veri önbellekte tekrar kullanılır.

    İki hızlandırma (ilk sürümümüz 200 iterasyonda bile yakınsamadı, çünkü
    embedding spektrumu düz: ardışık tekil değerler birbirine çok yakın ve
    yakınsama hızı sigma_{k+1}/sigma_k oranına bağlı):
      * oversample=p: k yerine k+p vektörle iterasyon. Hız oranı
        sigma_{k+p+1}/sigma_k'ya iyileşir (randomize SVD'deki fikir).
      * Rayleigh-Ritz: her adımda küçük (k+p)x(k+p) matris H = V^T G V'nin
        özvektörleriyle V'yi döndürürüz. Böylece alt uzayın İÇİNDEKİ en iyi
        tekil vektörleri seçeriz. (Küçük matris için np.linalg.eigh kullanıyoruz;
        asıl iş -- G @ V ve QR -- bizim iterasyonumuz.)
      Ölçülen etki (k=100): 200 iterasyon ve %0.2 hata -> 53 iterasyon ve 3e-6 hata.

    Adımlar:
      1. V: rastgele (d x (k+p)), QR ile ortonormal yap
      2. Tekrarla: V <- qr(G @ V); (Ritz) V <- V @ eig(V^T G V)
      3. Yakınsama: sigma^2 tahminleri (göreli) tol'den az değişince dur
      4. sigma_i = ||X v_i||,  u_i = X v_i / sigma_i, ilk k'yı tut

    Returns: U_k (N x k), S_k (k,), Vt_k (k x d)  [+ iterasyon sayısı]
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)                # hassasiyet: G'de kareler alındığı için float64
    G = X.T @ X                                        # d x d Gram matrisi (tek seferlik BLAS-3)
    m = min(k + oversample, X.shape[1])                # iterasyondaki vektör sayısı
    V, _ = np.linalg.qr(rng.standard_normal((X.shape[1], m)))
    lam_old = np.zeros(k)
    it = 0
    for it in range(1, max_iter + 1):
        Z = G @ V                                      # güç adımı: büyük özdeğer yönleri büyür
        V, _ = np.linalg.qr(Z)                         # ortonormalleştir: vektörler aynı yöne çökmesin
        if ritz:
            H = V.T @ (G @ V)                          # alt uzaya izdüşürülmüş küçük problem
            w, E = np.linalg.eigh(H)                   # artan sırada özdeğerler
            w, E = w[::-1], E[:, ::-1]                 # azalan sıraya çevir
            V = V @ E                                  # alt uzay içinde en iyi yönlere döndür
            lam = w[:k]                                # sigma^2 tahminleri
        else:
            lam = np.linalg.norm(Z, axis=0)[:k]        # ||G v|| ~ sigma^2 (Ritz'siz kaba tahmin)
        if np.max(np.abs(lam - lam_old) / np.maximum(np.abs(lam), 1e-12)) < tol:
            break
        lam_old = lam
    XV = X @ V                                         # N x m
    S = np.linalg.norm(XV, axis=0)                     # sigma_i = ||X v_i||
    order = np.argsort(-S)[:k]                         # büyükten küçüğe, ilk k
    S, V, XV = S[order], V[:, order], XV[:, order]
    U = XV / S                                         # u_i = X v_i / sigma_i
    out = (U, S, V.T)
    return (*out, it) if return_iters else out


def reconstruction_error(X, U, S, Vt):
    """||X - U diag(S) Vt||_F (Frobenius normu). Düşük = daha iyi yaklaşım."""
    X_recon = (U * S) @ Vt                             # U @ diag(S) yerine yayınlama: k x k matris kurmaz
    return float(np.linalg.norm(X - X_recon))


# --- Kütüphane yöntemlerini aynı arayüze sarıyoruz: (X, k) -> (X_recon) ---------

def run_full_svd(X, k):
    """Tam SVD: tüm 384 tekil değeri hesaplar, sonra ilk k'yı tutar (k'dan bağımsız maliyet)."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k]


def run_truncated(X, k):
    """Randomize Truncated SVD (Halko et al.): rastgele izdüşüm + birkaç güç iterasyonu."""
    m = TruncatedSVD(n_components=k, algorithm="randomized", random_state=SEED)
    Z = m.fit_transform(X)                             # = U * S
    return Z, m.components_


def run_pca(X, k):
    """PCA: veriyi ortalayıp (mean'i çıkarıp) SVD uygular -> kovaryansın özvektörleri."""
    m = PCA(n_components=k, random_state=SEED)
    Z = m.fit_transform(X)
    return Z, m


def recon_of(method, X, k, out):
    """Her yöntemin çıktısından yeniden oluşturma hatasını hesaplar."""
    if method in ("power_iteration", "full_svd"):
        return reconstruction_error(X, *out)
    if method == "truncated_svd":
        Z, comps = out
        return float(np.linalg.norm(X - Z @ comps))
    if method == "pca":
        Z, m = out
        return float(np.linalg.norm(X - m.inverse_transform(Z)))   # ortalama geri eklenir


METHODS = {
    "power_iteration": lambda X, k: power_iteration_svd(X, k),
    "full_svd": run_full_svd,
    "truncated_svd": run_truncated,
    "pca": run_pca,
}
LABELS = {"power_iteration": "Kendi güç iterasyonumuz", "full_svd": "np.linalg.svd (tam)",
          "truncated_svd": "TruncatedSVD (randomize)", "pca": "PCA"}
COLORS = {"power_iteration": C["orange"], "full_svd": C["red"],
          "truncated_svd": C["blue"], "pca": C["aqua"]}


def peak_memory_mb(func, *args):
    """Tek çalıştırmada tracemalloc ile tepe bellek (NumPy ayırmaları da izlenir)."""
    tracemalloc.start()
    func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024**2


THREADS = (1, 8)   # Tüm yöntemler iki ayarla ölçülür (adil: aynı ayar herkese)


def compare_svd_methods(X, k_values, runs=5, warmup=1):
    """
    Sabit matris üzerinde her yöntem x her k x her thread ayarı için süre, hata, bellek.

    Neden 1 thread de? Buradaki matrisler küçük (384 x k). Çok thread'li BLAS
    her küçük çağrıda thread'leri uyandırıp senkronize eder; iş birkaç
    mikrosaniyeyse bu sabit yük işin kendisinden büyük olur (Amdahl: seri
    kısım baskın). Ön denemede güç iterasyonumuz 1 thread'de 3-8 kat hızlıydı.
    """
    rows = []
    for th in THREADS:
        with threadpool_limits(th):
            for k in k_values:
                for name, fn in METHODS.items():
                    b = benchmark(fn, X, k, warmup=warmup, runs=runs, auto_runs=True)
                    err = recon_of(name, X, k, b["results"])
                    mem = peak_memory_mb(fn, X, k)
                    row = {"threads": th, "k": k, "method": name, "time_s": b["median"],
                           "raw_times": b["raw_times"], "recon_error": err,
                           "rel_error": err / np.linalg.norm(X), "peak_mem_mb": mem}
                    if name == "power_iteration":
                        row["iters"] = power_iteration_svd(X, k, return_iters=True)[3]
                    rows.append(row)
                    print(f"  th={th} k={k:3d} {name:16s} {b['median'] * 1e3:8.1f} ms  "
                          f"hata={err:8.3f}  bellek={mem:6.1f} MB"
                          + (f"  iter={row['iters']}" if "iters" in row else ""))
    return rows


def compare_sizes(C_all, sizes, k=50, runs=5, warmup=1):
    """Sabit k=50'de matris satır sayısı (N) arttıkça süre nasıl değişiyor? (her iki thread ayarı)"""
    rows = []
    for th in THREADS:
        with threadpool_limits(th):
            for n in sizes:
                X = np.asarray(C_all[:n], dtype=np.float64)
                for name, fn in METHODS.items():
                    b = benchmark(fn, X, k, warmup=warmup, runs=runs, auto_runs=True)
                    rows.append({"threads": th, "n": n, "k": k, "method": name,
                                 "time_s": b["median"], "raw_times": b["raw_times"]})
                    print(f"  th={th} N={n:6d} {name:16s} {b['median'] * 1e3:8.1f} ms")
    return rows


def recall_vs_k(C, Q, k_values):
    """
    İndirgenmiş uzayda arama kalitesi: TruncatedSVD ile k boyuta indir, normalize et,
    top-5 ara; tam 384 boyutlu kesin aramanın top-5'iyle örtüşme oranı (recall@5).
    Not: TruncatedSVD ortalama çıkarmaz -> cosine geometrisini (orijine göre açı) korur.
    """
    Cn, Qn = normalize_rows(C), normalize_rows(Q)
    exact = np.argsort(-(Qn @ Cn.T), axis=1)[:, :TOP_K]
    out = []
    for k in k_values:
        m = TruncatedSVD(n_components=k, random_state=SEED).fit(C)
        Ck = normalize_rows(m.transform(C))
        Qk = normalize_rows(m.transform(Q))
        approx = np.argsort(-(Qk @ Ck.T), axis=1)[:, :TOP_K]
        rec = np.mean([len(set(a) & set(e)) / TOP_K for a, e in zip(approx, exact)])
        out.append({"k": k, "recall_at_5": float(rec),
                    "explained_variance": float(m.explained_variance_ratio_.sum())})
        print(f"  k={k:3d} recall@5={rec:.3f}  açıklanan varyans={out[-1]['explained_variance']:.3f}")
    return out


def plot_svd(rows_k, rows_n, rec, save_path, n):
    """Figür 3: (a) süre-k, (b) süre-N  [düz = 1 thread, kesikli = 8 thread], (c) recall@5-k."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for m in METHODS:
        for th, ls, a in ((1, "o-", 1.0), (8, "--", 0.45)):
            pts = [(r["k"], r["time_s"] * 1e3) for r in rows_k if r["method"] == m and r["threads"] == th]
            ax.plot(*zip(*pts), ls, color=COLORS[m], alpha=a,
                    label=LABELS[m] if th == 1 else None, linewidth=2 if th == 1 else 1.3)
    ax.plot([], [], linestyle="--", marker="", color=plotstyle.MUTED, label="kesikli: 8 BLAS thread")
    ax.set_xlabel("Bileşen sayısı k")
    ax.set_ylabel("Süre (ms, medyan, log)")
    ax.set_yscale("log")
    ax.set_title(f"(a) Süre vs k  (N={n:,}, d=384)".replace(f"{n:,}", f"{n:,}".replace(",", ".")))
    ax.legend(fontsize=8)

    ax = axes[1]
    for m in METHODS:
        for th, ls, a in ((1, "o-", 1.0), (8, "--", 0.45)):
            pts = [(r["n"], r["time_s"] * 1e3) for r in rows_n if r["method"] == m and r["threads"] == th]
            ax.plot(*zip(*pts), ls, color=COLORS[m], alpha=a, linewidth=2 if th == 1 else 1.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matris satır sayısı N (d=384, k=50)")
    ax.set_ylabel("Süre (ms, medyan)")
    ax.set_title("(b) Süre vs matris boyutu")

    ax = axes[2]
    ks = [r["k"] for r in rec]
    ax.plot(ks, [r["recall_at_5"] for r in rec], "o-", color=C["blue"], label="recall@5 (arama kalitesi)")
    ax.plot(ks, [r["explained_variance"] for r in rec], "s--", color=C["aqua"],
            label="açıklanan varyans oranı")
    for r in rec:
        ax.annotate(f"{r['recall_at_5']:.2f}", (r["k"], r["recall_at_5"]), textcoords="offset points",
                    xytext=(6, -13), ha="left", fontsize=8, color=plotstyle.INK2)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Bileşen sayısı k")
    ax.set_title("(c) İndirgenmiş uzayda arama kalitesi")
    ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plotstyle.save(fig, save_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2B: Boyut İndirgeme")
    print("=" * 60)
    print(f"Thread sayısı (BLAS + Numba): {configure_threads()}")
    C_all, Q = load_embeddings()
    # Doğrulama, k taraması ve recall standart korpusta (tamamı); N taraması ayrıca aşağıda.
    X = np.asarray(C_all, dtype=np.float64)
    print(f"Matris: {X.shape}")

    # 1. Doğrulama: kendi SVD'miz tam SVD ile aynı tekil değerleri buluyor mu?
    k_test = 50
    t0 = time.perf_counter()
    U_pi, S_pi, Vt_pi, iters = power_iteration_svd(X, k_test, return_iters=True)
    t_pi = time.perf_counter() - t0
    U_np, S_np, Vt_np = np.linalg.svd(X, full_matrices=False)
    err_pi = reconstruction_error(X, U_pi, S_pi, Vt_pi)
    err_np = reconstruction_error(X, U_np[:, :k_test], S_np[:k_test], Vt_np[:k_test])
    sv_rel = float(np.max(np.abs(S_pi - S_np[:k_test]) / S_np[:k_test]))
    print(f"Doğrulama k={k_test}: hata kendi={err_pi:.4f}, numpy={err_np:.4f}, "
          f"tekil değer max göreli fark={sv_rel:.2e}, iterasyon={iters}, {t_pi * 1e3:.0f} ms")

    k_values = [10, 25, 50, 100, 200, 300]
    print(f"\nk = {k_values} için karşılaştırma...")
    rows_k = compare_svd_methods(X, k_values)

    print("\nMatris boyutuyla ölçekleme (k=50)...")
    rows_n = compare_sizes(C_all, scaling_sizes(C_all.shape[0]))

    print("\nİndirgenmiş uzayda recall@5...")
    rec = recall_vs_k(X, Q.astype(np.float64),
                      [16, 32, 64, 128, 192, 256])

    save_rows_csv(rows_k, RESULTS_DIR / "phase2b_svd_vs_k.csv")
    save_rows_csv(rows_n, RESULTS_DIR / "phase2b_svd_vs_n.csv")
    save_json({"validation": {"k": k_test, "err_power": err_pi, "err_numpy": err_np,
                              "max_rel_sv_diff": sv_rel, "iters": iters,
                              "singular_values_top10": S_np[:10]},
               "vs_k": rows_k, "vs_n": rows_n, "recall": rec,
               "frobenius_X": float(np.linalg.norm(X))}, "phase2b_svd.json")
    plot_svd(rows_k, rows_n, rec, FIGURES_DIR / "fig3_svd.png", X.shape[0])
