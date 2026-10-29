"""
Phase 2A: Sparse vs Dense -- Temsil kararı (starter dosyası, tamamlandı)

Aynı arama işlemi S = Q @ X.T (sorgular x dokümanlar skor matrisi) dört
farklı temsil kombinasyonunda, KENDİ yazdığımız çekirdeklerle hesaplanır:

  SS  : sparse TF-IDF sorgu  x  sparse TF-IDF korpus   (ters indeks / inverted index)
  SD  : sparse TF-IDF sorgu  x  dense  TF-IDF korpus   (aynı veri, yoğun saklanmış)
  DDt : dense  TF-IDF sorgu  x  dense  TF-IDF korpus   (aynı veri, iki taraf yoğun)
  DDe : dense  MiniLM sorgu  x  dense  MiniLM korpus   (384 boyut, pratik dense seçenek)

Her biri için kütüphane referansı (scipy / NumPy-BLAS) da ölçülür ve
sonuçların referansla aynı olduğu doğrulanır.

Neden Numba? Kendi döngülerimizi saf Python'da yazsak her kombinasyon
dakikalar sürer ve ölçtüğümüz şey temsil değil yorumlayıcı yükü olur.
Numba ile derlenmiş döngüler, temsillerin *kendisinin* maliyetini adil kıyaslar.
Tüm çekirdekler sorgular üzerinde prange ile paraleldir -> karşılaştırma eşit koşulda.

Kavramlar: bellek bant genişliği (sıfırları okumak da zaman alır),
önbellek/bellek yerleşimi (satır vs sütun erişimi), BLAS.
"""

import sys
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import benchmark, save_rows_csv  # noqa: E402
from common import (FIGURES_DIR, RESULTS_DIR, configure_threads, load_embeddings,  # noqa: E402
                    load_tfidf, save_json, scaling_sizes)
import plotstyle  # noqa: E402
from plotstyle import C, plt  # noqa: E402


def memory_usage(array_or_matrix):
    """
    Bir matrisin bellekte kapladığı yer (MB).
    CSR için üç dizinin toplamı: data (değerler), indices (sütun no), indptr (satır başları).
    """
    if sparse.issparse(array_or_matrix):
        return (array_or_matrix.data.nbytes +
                array_or_matrix.indices.nbytes +
                array_or_matrix.indptr.nbytes) / (1024 * 1024)
    return array_or_matrix.nbytes / (1024 * 1024)


# ==========================================================================
# KENDİ ÇARPIM ÇEKİRDEKLERİMİZ
# ==========================================================================

@njit(parallel=True, cache=True)
def _ss_kernel(q_indptr, q_indices, q_data, x_indptr, x_indices, x_data, n_docs):
    """
    Sparse x sparse: ters indeks (inverted index) yöntemi.

    X'i CSC formatında veriyoruz: CSC'de her SÜTUN (= kelime) için o kelimeyi
    içeren dokümanların listesi ardışık durur -> bu tam olarak arama
    motorlarının "posting list"idir. Sorgudaki her kelime için sadece o
    kelimeyi içeren dokümanları gezer, skorlarına katkı ekleriz.
    İş miktarı: sum_{t in sorgu} |posting(t)|  <<  N x V. Sıfırlara hiç dokunulmaz.
    """
    nq = q_indptr.shape[0] - 1
    out = np.zeros((nq, n_docs), dtype=np.float32)
    for i in prange(nq):                                   # her sorgu ayrı thread'de
        for p in range(q_indptr[i], q_indptr[i + 1]):      # sorgunun sıfır olmayan kelimeleri
            t = q_indices[p]
            w = q_data[p]
            for r in range(x_indptr[t], x_indptr[t + 1]):  # t kelimesinin posting listesi
                out[i, x_indices[r]] += w * x_data[r]      # dağınık (scatter) yazma
    return out


@njit(parallel=True, cache=True)
def _sd_kernel(q_indptr, q_indices, q_data, XT):
    """
    Sparse x dense: sorgunun sıfır olmayan her kelimesi için dense korpusun
    o kelimeye ait SATIRINI (XT[t], uzunluk N) skora ekler.

    Bellek yerleşimi kararı: korpusu X (N x V) yerine XT (V x N) olarak
    saklıyoruz. Böylece "t kelimesinin tüm dokümanlardaki ağırlığı" bellekte
    ARDIŞIK durur; X[:, t] gibi V adımlı (strided) erişim olsaydı her eleman
    ayrı bir önbellek satırı (64 B) çekerdi -> bant genişliği israfı.
    """
    nq = q_indptr.shape[0] - 1
    n_docs = XT.shape[1]
    out = np.zeros((nq, n_docs), dtype=np.float32)
    for i in prange(nq):
        for p in range(q_indptr[i], q_indptr[i + 1]):
            t = q_indices[p]
            w = q_data[p]
            row = XT[t]
            for j in range(n_docs):                        # ardışık, SIMD'e uygun döngü
                out[i, j] += w * row[j]
    return out


@njit(parallel=True, cache=True)
def _dd_kernel(Q, X):
    """
    Dense x dense: klasik üçlü döngü, out[i, j] = sum_t Q[i, t] * X[j, t].
    Her iki matris de satır öncelikli -> iç döngü (t) her ikisinde de ardışık okur.
    Sıfır olsa da olmasa da her elemanı çarpar: TF-IDF için işin %99'u 0*0.
    """
    nq, d = Q.shape
    n_docs = X.shape[0]
    out = np.zeros((nq, n_docs), dtype=np.float32)
    for i in prange(nq):
        for j in range(n_docs):
            s = np.float32(0.0)
            for t in range(d):
                s += Q[i, t] * X[j, t]
            out[i, j] = s
    return out


def sparse_dot_sparse(A_sparse, B_csc):
    """A (CSR, q x V) @ B.T, B önceden CSC'ye çevrilmiş (ters indeks)."""
    return _ss_kernel(A_sparse.indptr, A_sparse.indices, A_sparse.data,
                      B_csc.indptr, B_csc.indices, B_csc.data, B_csc.shape[0])


def sparse_dot_dense(A_sparse, BT_dense):
    """A (CSR, q x V) @ B.T, B'nin transpozu (V x N) ardışık olarak verilir."""
    return _sd_kernel(A_sparse.indptr, A_sparse.indices, A_sparse.data, BT_dense)


def dense_dot_dense(A_dense, B_dense):
    """A (q x d) @ B.T (N x d), kendi üçlü döngümüz."""
    return _dd_kernel(A_dense, B_dense)


# ==========================================================================
# KARŞILAŞTIRMA
# ==========================================================================

def compare_representations(sizes, runs=5, warmup=2):
    """
    Her korpus boyutunda bellek + çarpım süresini ölçer.
    Dönüş: her boyut için bir sözlük (bellek MB, süreler, doğruluk).
    """
    X_all, Qs = load_tfidf()
    C_all, Qe = load_embeddings()
    Qs = Qs.astype(np.float32)
    Qd = Qs.toarray()                                   # sorgunun dense TF-IDF hali
    rows = []

    for n in sizes:
        print(f"\n--- Korpus boyutu: {n} ---")
        X = X_all[:n].astype(np.float32)
        X_csc = X.tocsc()                               # ters indeks yapısı (posting listeleri)
        X_dense = X.toarray()                           # AYNI veri, sıfırlarıyla birlikte
        XT_dense = np.ascontiguousarray(X_dense.T)      # SD çekirdeği için kelime-öncelikli yerleşim
        E = np.ascontiguousarray(C_all[:n])
        density = X.nnz / (X.shape[0] * X.shape[1])
        print(f"TF-IDF: {X.shape}, nnz={X.nnz}, yoğunluk={density:.4%}")

        r = {
            "corpus_size": n, "vocab": X.shape[1], "nnz": X.nnz, "density": density,
            "mem_sparse_tfidf_mb": memory_usage(X),
            "mem_dense_tfidf_mb": memory_usage(X_dense),
            "mem_dense_minilm_mb": memory_usage(E),
        }
        print(f"Bellek: CSR={r['mem_sparse_tfidf_mb']:.2f} MB, dense TF-IDF="
              f"{r['mem_dense_tfidf_mb']:.1f} MB, MiniLM={r['mem_dense_minilm_mb']:.2f} MB")

        # (ad, kendi fonksiyonumuz, argümanlar, referans fonksiyon, referans argümanları)
        cases = [
            ("SS", sparse_dot_sparse, (Qs, X_csc), lambda a, b: (a @ b.T).toarray(), (Qs, X)),
            ("SD", sparse_dot_dense, (Qs, XT_dense), lambda a, b: a @ b, (Qs, XT_dense)),
            ("DDt", dense_dot_dense, (Qd, X_dense), lambda a, b: a @ b.T, (Qd, X_dense)),
            ("DDe", dense_dot_dense, (Qe, E), lambda a, b: a @ b.T, (Qe, E)),
        ]
        for name, own, own_args, ref, ref_args in cases:
            b_own = benchmark(own, *own_args, warmup=warmup, runs=runs, auto_runs=True)
            b_ref = benchmark(ref, *ref_args, warmup=warmup, runs=runs, auto_runs=True)
            ok = bool(np.allclose(b_own["results"], np.asarray(b_ref["results"]), atol=1e-5))
            r[f"t_{name}_own"] = b_own["median"]
            r[f"t_{name}_ref"] = b_ref["median"]
            r[f"raw_{name}_own"] = b_own["raw_times"]
            r[f"ok_{name}"] = ok
            print(f"  {name:4s} kendi={b_own['median'] * 1e3:8.2f} ms | "
                  f"kütüphane={b_ref['median'] * 1e3:8.2f} ms | doğru={ok}")
        rows.append(r)
    return rows


def plot_sparse_dense(rows, save_path):
    """
    Figür 2: sol = bellek (log ölçek), sağ = çarpım süresi (kendi çekirdeklerimiz).
    İki ölçü farklı birimde olduğu için çift eksen yerine iki ayrı panel.
    """
    sizes = [r["corpus_size"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    series = [("mem_dense_tfidf_mb", "Dense TF-IDF (aynı veri, sıfırlarla)", C["red"]),
              ("mem_dense_minilm_mb", "Dense MiniLM (384 boyut)", C["blue"]),
              ("mem_sparse_tfidf_mb", "Sparse TF-IDF (CSR)", C["aqua"])]
    for key, label, col in series:
        ys = [r[key] for r in rows]
        ax.plot(sizes, ys, "o-", color=col, label=label)
        ax.annotate(f"{ys[-1]:.1f} MB", (sizes[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8.5, color=plotstyle.INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Korpus boyutu (doküman)")
    ax.set_ylabel("Bellek (MB, log)")
    ax.set_title("Bellek ayak izi")
    ax.legend(loc="upper left")

    ax = axes[1]
    tseries = [("DDt", "Dense x Dense (TF-IDF, V boyut)", C["red"]),
               ("SD", "Sparse x Dense (TF-IDF)", C["yellow"]),
               ("DDe", "Dense x Dense (MiniLM, 384)", C["blue"]),
               ("SS", "Sparse x Sparse (ters indeks)", C["aqua"])]
    for key, label, col in tseries:
        ys = [r[f"t_{key}_own"] * 1e3 for r in rows]
        ax.plot(sizes, ys, "o-", color=col, label=label)
        ref = [r[f"t_{key}_ref"] * 1e3 for r in rows]
        ax.plot(sizes, ref, "--", color=col, alpha=0.55, linewidth=1.3)
    ax.plot([], [], linestyle="--", marker="", color=plotstyle.MUTED,
            label="kesikli: kütüphane (scipy / BLAS)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Korpus boyutu (doküman)")
    ax.set_ylabel("100 sorgu için süre (ms, log, medyan)")
    ax.set_title("Skor matrisi Q @ X.T hesaplama süresi")
    ax.legend(loc="upper left", fontsize=8.3)

    plt.tight_layout()
    plotstyle.save(fig, save_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2A: Sparse vs Dense Karşılaştırması")
    print("=" * 60)
    print(f"Thread sayısı (BLAS + Numba): {configure_threads()}")
    n_total = load_embeddings()[0].shape[0]
    sizes = scaling_sizes(n_total)
    print(f"Seçilen boyutlar: {sizes}")

    rows = compare_representations(sizes)
    save_rows_csv(rows, RESULTS_DIR / "phase2a_sparse_dense.csv")
    save_json(rows, "phase2a_sparse_dense.json")
    plot_sparse_dense(rows, FIGURES_DIR / "fig2_sparse_dense.png")
