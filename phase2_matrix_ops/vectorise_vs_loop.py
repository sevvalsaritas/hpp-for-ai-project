"""
Phase 2C: Vektörizasyon vs Eleman-Eleman vs JIT (starter dosyası, tamamlandı)

Ana işlem: arama çekirdeği olan COSINE BENZERLİK MATRİSİ  S = Qn @ Cn.T
(Qn, Cn satır-normalize -> cosine = nokta çarpım). Dört (+1) uygulama:
  1. Saf Python döngüleri (list of lists)
  2. NumPy vektörize (Qn @ Cn.T -> BLAS sgemm)
  3. Numba @njit (aynı döngü, derlenmiş, tek çekirdek)
  4. Numba @njit(parallel=True) + prange (dış döngü çekirdeklere dağıtılır)
  5. (ek) Numba parallel + fastmath: toplama sırasını değiştirmeye izin verir -> SIMD

Ek olarak starter'daki Gram matrisi X^T X de dört şekilde uygulanır (ikinci işlem).

Anahtar soru: Numba ne zaman NumPy'yi yener, ne zaman kaybeder?
Kavramlar: yorumlayıcı yükü, SIMD, BLAS'ın önbellek bloklama (cache blocking),
bellek bant genişliği, Amdahl (çağrı yükü küçük boyutta seri kısımdır).
"""

import sys
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import benchmark, save_rows_csv  # noqa: E402
from common import (FIGURES_DIR, RESULTS_DIR, configure_threads, load_embeddings,  # noqa: E402
                    normalize_rows, save_json)
import plotstyle  # noqa: E402
from plotstyle import METHOD_COLORS, plt  # noqa: E402


# ==========================================================================
# 1) COSINE BENZERLİK MATRİSİ
# ==========================================================================

def cosine_python(Q, C):
    """
    Saf Python: Q ve C Python listeleri. Her (i, j) için d uzunluğunda iç döngü.
    Her çarpma-toplama yorumlayıcıda bytecode + float nesnesi oluşturma demek.
    """
    out = []
    for q in Q:
        row = []
        for c in C:
            s = 0.0
            for a, b in zip(q, c):
                s += a * b
            row.append(s)
        out.append(row)
    return out


def cosine_numpy(Q, C):
    """
    NumPy: tek satır. Arkada BLAS sgemm çalışır:
      - önbellek bloklama: veriyi L1/L2'ye sığan parçalara bölüp tekrar kullanır
      - SIMD (AVX2) + FMA: tek komutla 8 float çarp-topla
      - çoklu thread: GIL serbest, tüm çekirdekler
    """
    return Q @ C.T


@njit(cache=True)
def cosine_numba(Q, C):
    """Numba seri: Python döngüsüyle AYNI algoritma, makine koduna derlenmiş."""
    nq, d = Q.shape
    n = C.shape[0]
    out = np.empty((nq, n), dtype=np.float32)
    for i in range(nq):
        for j in range(n):
            s = np.float32(0.0)
            for t in range(d):
                s += Q[i, t] * C[j, t]
            out[i, j] = s
    return out


@njit(parallel=True, cache=True)
def cosine_numba_parallel(Q, C):
    """
    Numba paralel: dış döngü prange ile thread'lere bölünür. Her thread farklı
    out satırlarına yazar -> yarış koşulu (race condition) yok, kilit gerekmez.
    Neden dış döngü? İş parçası büyük olur, thread başlatma yükü amorti edilir.
    """
    nq, d = Q.shape
    n = C.shape[0]
    out = np.empty((nq, n), dtype=np.float32)
    for i in prange(nq):
        for j in range(n):
            s = np.float32(0.0)
            for t in range(d):
                s += Q[i, t] * C[j, t]
            out[i, j] = s
    return out


@njit(parallel=True, fastmath=True, cache=True)
def cosine_numba_parallel_fm(Q, C):
    """
    fastmath=True: derleyiciye "toplama sırasını değiştirebilirsin" izni verir.
    Kayan noktada (a+b)+c != a+(b+c) olduğu için normalde derleyici iç döngüdeki
    toplamı vektörleştiremez; izin verince 8'li SIMD kısmi toplamlar kullanır.
    Bedeli: sonuç son basamaklarda (~1e-7) farklı olabilir -> tolerans 1e-5 içinde.
    Ek olarak iş bölümü (i, j) çiftleri üzerinden: sorgu sayısı az olsa bile
    tüm çekirdekler meşgul olur.
    """
    nq, d = Q.shape
    n = C.shape[0]
    out = np.empty((nq, n), dtype=np.float32)
    for ij in prange(nq * n):
        i = ij // n
        j = ij - i * n
        s = np.float32(0.0)
        for t in range(d):
            s += Q[i, t] * C[j, t]
        out[i, j] = s
    return out


# ==========================================================================
# 2) GRAM MATRİSİ  G = X^T X   (starter'daki işlem)
# ==========================================================================

def gram_elementwise(X):
    """Saf Python üçlü döngü (X: list of lists, m x n). Starter'daki temel sürüm."""
    m, n = len(X), len(X[0])
    G = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += X[k][i] * X[k][j]
            G[i][j] = s
    return G


def gram_vectorised(X):
    """NumPy tek satır: BLAS'a devredilir."""
    return X.T @ X


@njit(cache=True)
def gram_numba_serial(X):
    """
    Numba seri. Döngü sırası (k, i, j): X[k, :] satırı ardışık okunur -> önbellek dostu.
    Starter'daki (i, j, k) sırası X[k, i]'ye sütun yönünde (adımlı) erişirdi.
    """
    m, n = X.shape
    G = np.zeros((n, n), dtype=X.dtype)
    for k in range(m):
        for i in range(n):
            xi = X[k, i]
            for j in range(n):
                G[i, j] += xi * X[k, j]
    return G


@njit(parallel=True, cache=True)
def gram_numba_parallel(X):
    """
    Numba paralel: G'nin satırları (i) thread'lere bölünür. (k, i, j) sırası
    paralelde yarış koşulu yaratırdı (aynı G[i, j]'ye farklı k'lardan yazım);
    bu yüzden i dışarıda: her thread kendi G satırının tek sahibi.
    """
    m, n = X.shape
    G = np.zeros((n, n), dtype=X.dtype)
    for i in prange(n):
        for k in range(m):
            xi = X[k, i]
            for j in range(n):
                G[i, j] += xi * X[k, j]
    return G


# ==========================================================================
# BENCHMARK
# ==========================================================================

def verify_correctness(Qn, Cn):
    """Tüm uygulamalar NumPy ile aynı sonucu veriyor mu? (tolerans 1e-5)"""
    q, c = Qn[:5], Cn[:200]
    ref = cosine_numpy(q, c)
    checks = {
        "python": np.array(cosine_python(q.tolist(), c.tolist()), dtype=np.float32),
        "numba": cosine_numba(q, c),
        "numba_parallel": cosine_numba_parallel(q, c),
        "numba_parallel_fastmath": cosine_numba_parallel_fm(q, c),
    }
    res = {k: float(np.max(np.abs(v - ref))) for k, v in checks.items()}
    X = Cn[:300, :64].astype(np.float64)
    G = gram_vectorised(X)
    res["gram_python"] = float(np.max(np.abs(np.array(gram_elementwise(X.tolist())) - G)))
    res["gram_numba"] = float(np.max(np.abs(gram_numba_serial(X) - G)))
    res["gram_numba_parallel"] = float(np.max(np.abs(gram_numba_parallel(X) - G)))
    for k, v in res.items():
        print(f"  {k:26s} max |fark| = {v:.2e}  {'OK' if v < 1e-5 else 'HATA'}")
    assert all(v < 1e-5 for v in res.values())
    return res


def bench_cosine_sizes(Qn, Cn, configs, python_max_ops=1e9):
    """
    Farklı (sorgu sayısı, korpus boyutu) çiftlerinde tüm uygulamaları ölçer.
    Saf Python standart iş yükü dahil tüm noktalarda ölçülür (100 x 19.671 x 384
    = 7,6e8 çarp-topla, ölçüm başına ~20-30 s); sınır sadece kazara çok büyük
    bir boyut verilirse saatlerce beklemeyi önlemek için.
    """
    rows = []
    for nq, n in configs:
        q = np.ascontiguousarray(Qn[:nq])
        c = np.ascontiguousarray(Cn[:n])
        ops = nq * n * q.shape[1]
        row = {"n_queries": nq, "n_docs": n, "mult_adds": ops}
        impls = [("numpy", cosine_numpy, (q, c)),
                 ("numba", cosine_numba, (q, c)),
                 ("numba_parallel", cosine_numba_parallel, (q, c)),
                 ("numba_parallel_fastmath", cosine_numba_parallel_fm, (q, c))]
        if ops <= python_max_ops:
            impls.insert(0, ("python", cosine_python, (q.tolist(), c.tolist())))
        for name, fn, args in impls:
            # Saf Python için 1 warmup yeterli (JIT yok); diğerlerinde 3 (Numba derlemesi + cache)
            b = benchmark(fn, *args, warmup=1 if name == "python" else 3, runs=5, auto_runs=True)
            row[f"t_{name}"] = b["median"]
            row[f"raw_{name}"] = b["raw_times"]
            # GFLOP/s: her çarp-topla 2 işlem. Roofline analizinde de kullanılıyor.
            row[f"gflops_{name}"] = 2 * ops / b["median"] / 1e9
        rows.append(row)
        msg = " | ".join(f"{k[2:]}={row[k] * 1e3:.3f}ms" for k in row if k.startswith("t_"))
        print(f"  {nq:4d}x{n:6d}: {msg}")
    return rows


def bench_gram(Cn, m=2000, n=128):
    """Gram matrisi X^T X (m x n) için dört uygulama."""
    X = np.ascontiguousarray(Cn[:m, :n].astype(np.float64))
    row = {"m": m, "n": n}
    for name, fn, args in [("python", gram_elementwise, (X.tolist(),)),
                           ("numpy", gram_vectorised, (X,)),
                           ("numba", gram_numba_serial, (X,)),
                           ("numba_parallel", gram_numba_parallel, (X,))]:
        b = benchmark(fn, *args, warmup=1 if name == "python" else 3, runs=5, auto_runs=True)
        row[f"t_{name}"] = b["median"]
        row[f"raw_{name}"] = b["raw_times"]
        print(f"  Gram {m}x{n} {name:15s} {b['median'] * 1e3:10.3f} ms")
    return row


def plot_vectorisation(rows, gram_row, save_path):
    """
    Figür 4: (a) Python'a göre hızlanma (Python'un ölçüldüğü en büyük boyutta),
             (b) boyut büyüdükçe NumPy vs Numba süreleri (kim ne zaman kazanır).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1, 1.2]})

    ref = [r for r in rows if "t_python" in r][-1]
    names = ["python", "numba", "numba_parallel", "numba_parallel_fastmath", "numpy"]
    labels = ["Saf Python", "Numba @njit", "Numba prange", "Numba prange\n+fastmath", "NumPy (BLAS)"]
    cols = [METHOD_COLORS["python"], METHOD_COLORS["numba"], METHOD_COLORS["numba_parallel"],
            plotstyle.C["green"], METHOD_COLORS["numpy"]]
    sp = [ref["t_python"] / ref[f"t_{n}"] for n in names]
    ax = axes[0]
    bars = ax.bar(labels, sp, color=cols, width=0.62)
    ax.set_yscale("log")
    for b, s, n in zip(bars, sp, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.12,
                f"{s:,.0f}x\n{ref[f't_{n}'] * 1e3:.2f} ms", ha="center", fontsize=8.5,
                color=plotstyle.INK)
    ax.set_ylim(0.5, max(sp) * 8)
    ax.set_ylabel("Saf Python'a göre hızlanma (log)")
    n_tr = f"{ref['n_docs']:,}".replace(",", ".")     # Türkçe binlik ayıracı
    ax.set_title(f"(a) Cosine matrisi {ref['n_queries']} sorgu x {n_tr} doküman x 384")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    rows = sorted(rows, key=lambda r: r["mult_adds"])   # iki tarama ekseni iç içe: iş miktarına göre diz
    x = [r["mult_adds"] for r in rows]
    for n, lab, col in zip(names[1:], labels[1:], cols[1:]):
        ax.plot(x, [r[f"t_{n}"] * 1e3 for r in rows], "o-", color=col, label=lab.replace("\n", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("İş miktarı (sorgu x doküman x boyut, çarp-topla)")
    ax.set_ylabel("Süre (ms, medyan, log)")
    ax.set_title("(b) Boyut büyüdükçe: Numba küçükte, BLAS büyükte")
    ax.legend(fontsize=8.5)
    for r in rows:
        ax.annotate(f"{r['n_queries']}x{r['n_docs']}", (r["mult_adds"], r["t_numpy"] * 1e3),
                    textcoords="offset points", xytext=(0, -13), ha="center", fontsize=7,
                    color=plotstyle.MUTED)

    plt.tight_layout()
    plotstyle.save(fig, save_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2C: Vektörizasyon vs Eleman-Eleman vs JIT")
    print("=" * 60)
    print(f"Thread sayısı (BLAS + Numba): {configure_threads()}")
    C_all, Q = load_embeddings()
    Cn =normalize_rows(C_all).astype(np.float32)
    Qn = normalize_rows(Q).astype(np.float32)

    print("\nDoğruluk kontrolü...")
    corr = verify_correctness(Qn, Cn)

    n_full = Cn.shape[0]
    # İki tarama ekseni, standart iş yükünde (100 x tüm korpus) kesişir:
    #   sorgu sayısı 1 -> 10 -> 100 (korpus sabit = tamamı): GEMV'den GEMM'e geçiş,
    #   doküman sayısı 1K -> 5K -> tamamı (sorgu sabit = 100): iş büyüklüğü etkisi.
    configs = [(1, n_full), (10, n_full), (100, 1000), (100, 5000), (100, n_full)]
    print("\nCosine matrisi, boyut taraması...")
    rows = bench_cosine_sizes(Qn, Cn, configs)

    print("\nGram matrisi X^T X...")
    gram_row = bench_gram(Cn)

    save_rows_csv(rows, RESULTS_DIR / "phase2c_vectorise.csv")
    save_json({"correctness_max_abs_diff": corr, "cosine": rows, "gram": gram_row},
              "phase2c_vectorise.json")
    plot_vectorisation(rows, gram_row, FIGURES_DIR / "fig4_vectorisation.png")
