"""
Bonus: Yaklaşık En Yakın Komşu (ANN) -- kendi LSH ve IVF uygulamalarımız vs kesin arama.

Soru: Kesin (brute-force) arama her sorguda TÜM korpusu okur: O(N * d).
ANN yöntemleri korpusun sadece küçük bir kısmına bakarak hız kazanır, bedeli
bazı gerçek komşuları kaçırmaktır (recall < 1). Bizim ölçeğimizde buna değer mi?

  * LSH (random hyperplane / SimHash): b rastgele hiper-düzlem; her vektörün
    hangi tarafta olduğu b bitlik bir koda dönüşür. Cosine'i yakın vektörler
    aynı kodu alma eğilimindedir. L ayrı tablo -> kaçırma olasılığı düşer.
  * IVF (inverted file): korpus k-means ile n_list kümeye bölünür; sorgu en
    yakın nprobe kümenin dokümanlarına bakar (FAISS IVF'nin temel fikri).

Metrikler: recall@5 (kesin top-5 ile örtüşme), tek sorgu gecikmesi (medyan),
taranan aday oranı. Kavram: bellek bant genişliği -- tek sorguda kesin arama
30 MB okur; IVF bunun yalnızca nprobe/n_list kadarını.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase3_parallelism"))
from benchmark_utils import benchmark, save_rows_csv  # noqa: E402
from common import (FIGURES_DIR, RESULTS_DIR, SEED, TOP_K, configure_threads,  # noqa: E402
                    load_embeddings, normalize_rows, save_json)
import plotstyle  # noqa: E402
from plotstyle import C, plt  # noqa: E402
from search_kernels import search_block  # noqa: E402


def rerank(q, Cn, cand, k):
    """Aday kümesini kesin cosine ile yeniden sıralar (adaylar arasında kesin top-k)."""
    if len(cand) == 0:
        return np.full(k, -1), np.zeros(k, np.float32)
    s = Cn[cand] @ q                       # sadece adaylar: gather + GEMV
    kk = min(k, len(cand))
    top = np.argpartition(-s, kk - 1)[:kk]
    top = top[np.argsort(-s[top])]
    return cand[top], s[top]


class LSHIndex:
    """Random hyperplane LSH: L tablo x b bit."""

    def __init__(self, Cn, n_bits=12, n_tables=8, seed=SEED):
        rng = np.random.default_rng(seed)
        self.Cn = Cn
        # Merkezleme: MiniLM embedding'leri dar bir koni içinde toplanır (anizotropi).
        # Orijinden geçen rastgele hiper-düzlemlerin çoğu tüm noktaları aynı tarafta
        # bırakır ve kovalar ayrışmaz. Ortalamayı çıkarınca düzlemler verinin ortasından geçer.        self.mean = Cn.mean(axis=0)
        Xc = Cn - self.mean
        self.planes = rng.standard_normal((n_tables, Cn.shape[1], n_bits)).astype(np.float32)
        self.weights = (1 << np.arange(n_bits)).astype(np.int64)   # bitleri tamsayı koda çevirmek için
        self.tables = []
        for t in range(n_tables):
            codes = ((Xc @ self.planes[t]) > 0).astype(np.int64) @ self.weights
            buckets = defaultdict(list)
            for i, c in enumerate(codes):
                buckets[int(c)].append(i)
            self.tables.append({c: np.array(v) for c, v in buckets.items()})

    def query(self, q, k=TOP_K):
        """Her tabloda sorgunun kovasındaki dokümanları topla, birleşimini kesin sırala."""
        cands = []
        qc = q - self.mean                                 # korpusla aynı merkezleme
        for t, table in enumerate(self.tables):
            c = int(((qc @ self.planes[t]) > 0).astype(np.int64) @ self.weights)
            if c in table:
                cands.append(table[c])
        cand = np.unique(np.concatenate(cands)) if cands else np.array([], dtype=np.int64)
        return (*rerank(q, self.Cn, cand, k), len(cand))


def kmeans(X, n_clusters, n_iter=20, seed=SEED):
    """
    Basit (spherical) k-means: normalize vektörlerde cosine'e göre kümeleme.
    Atama adımı tek bir BLAS çağrısı (X @ merkezler.T) -> hızlı.
    """
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), n_clusters, replace=False)].copy()
    for _ in range(n_iter):
        assign = np.argmax(X @ cent.T, axis=1)
        for c in range(n_clusters):
            m = X[assign == c]
            if len(m):
                cent[c] = m.mean(axis=0)
        cent = normalize_rows(cent).astype(np.float32)
    return cent, np.argmax(X @ cent.T, axis=1)


class IVFIndex:
    """Inverted file index: k-means kümeleri + küme başına doküman listesi."""

    def __init__(self, Cn, n_list=None):
        self.Cn = Cn
        n_list = n_list or int(np.sqrt(len(Cn)))      # yaygın kural: n_list ~ sqrt(N)
        self.cent, assign = kmeans(Cn, n_list)
        self.lists = [np.where(assign == c)[0] for c in range(n_list)]

    def query(self, q, k=TOP_K, nprobe=4):
        """En yakın nprobe kümenin dokümanlarını aday al, kesin sırala."""
        cs = self.cent @ q
        probe = np.argpartition(-cs, nprobe - 1)[:nprobe]
        cand = np.concatenate([self.lists[c] for c in probe])
        return (*rerank(q, self.Cn, cand, k), len(cand))


def eval_index(name, params, query_fn, Qn, exact_idx, n_docs):
    """Recall@5, tek sorgu gecikmesi (medyan) ve taranan aday oranını ölçer."""
    def run_all():
        return [query_fn(q) for q in Qn]
    b = benchmark(run_all, warmup=2, runs=5)
    res = b["results"]
    rec = np.mean([len(set(r[0]) & set(e)) / TOP_K for r, e in zip(res, exact_idx)])
    frac = np.mean([r[2] for r in res]) / n_docs
    row = {"method": name, **params, "recall_at_5": float(rec), "per_query_ms": b["median"] / len(Qn) * 1e3,
           "candidate_fraction": float(frac), "raw_times": b["raw_times"]}
    print(f"  {name:6s} {params}: recall@5={rec:.3f}  {row['per_query_ms']:.3f} ms/sorgu  aday={frac:.1%}")
    return row


def main():
    configure_threads()
    C_all, Q = load_embeddings()
    Cn = np.ascontiguousarray(normalize_rows(C_all).astype(np.float32))
    Qn = np.ascontiguousarray(normalize_rows(Q).astype(np.float32))
    n = Cn.shape[0]
    exact_idx, _ = search_block(Qn, Cn, TOP_K)

    # Kesin arama referansları: tek tek sorgu (GEMV, bant genişliği sınırlı) ve toplu (GEMM)
    b1 = benchmark(lambda: [search_block(q[None], Cn, TOP_K) for q in Qn], warmup=2, runs=5)
    bb = benchmark(search_block, Qn, Cn, TOP_K, warmup=3, runs=5, auto_runs=True)
    rows = [{"method": "exact", "mode": "tek tek", "recall_at_5": 1.0,
             "per_query_ms": b1["median"] / len(Qn) * 1e3, "candidate_fraction": 1.0},
            {"method": "exact", "mode": "toplu 100", "recall_at_5": 1.0,
             "per_query_ms": bb["median"] / len(Qn) * 1e3, "candidate_fraction": 1.0}]
    print(f"Kesin: tek tek {rows[0]['per_query_ms']:.3f} ms/sorgu, toplu {rows[1]['per_query_ms']:.4f} ms/sorgu")

    build = {}
    # Neden az bit / çok tablo? Benzer makalelerin cosine'i ~0.6 (açı ~53°). İki vektörün
    # tek bir hiper-düzlemin aynı tarafına düşme olasılığı 1 - açı/180° ~ 0.70; b bitte 0.70^b
    # (8 bitte 0.06!). L tabloyla en az birinde çarpışma 1 - (1 - 0.70^b)^L. Az bit = büyük
    # kovalar (çok aday), çok tablo = yüksek recall ama daha çok iş: denge eğrisini tarıyoruz.
    for bits, tables in [(4, 8), (6, 8), (6, 16), (8, 16), (8, 32)]:
        t0 = time.perf_counter()
        idx = LSHIndex(Cn, bits, tables)
        build[f"lsh_{bits}_{tables}"] = time.perf_counter() - t0
        rows.append(eval_index("LSH", {"bits": bits, "tables": tables}, idx.query, Qn, exact_idx, n))

    t0 = time.perf_counter()
    ivf = IVFIndex(Cn)
    build["ivf"] = time.perf_counter() - t0
    for nprobe in [1, 2, 4, 8, 16, 32]:
        rows.append(eval_index("IVF", {"n_list": len(ivf.lists), "nprobe": nprobe},
                               lambda q, p=nprobe: ivf.query(q, TOP_K, p), Qn, exact_idx, n))

    save_rows_csv(rows, RESULTS_DIR / "bonus_ann.csv")
    save_json({"n_docs": n, "rows": rows, "build_time_s": build}, "bonus_ann.json")
    plot(rows, FIGURES_DIR / "figA_ann.png", n)


def plot(rows, save_path, n):
    """Ek figür: recall@5 vs sorgu başına süre (sol üst köşe = iyi)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, col, mk in (("LSH", C["orange"], "s"), ("IVF", C["aqua"], "o")):
        pts = [(r["per_query_ms"], r["recall_at_5"], r) for r in rows if r["method"] == m]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=col, label=f"{m} (kendi uygulamamız)")
        for x, y, r in pts:
            lab = f"np={r['nprobe']}" if m == "IVF" else f"{r['bits']}b x{r['tables']}"
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=7.5,
                        color=plotstyle.MUTED)
    ex = [r for r in rows if r["method"] == "exact"]
    for r, col in zip(ex, (C["red"], C["blue"])):
        ax.scatter([r["per_query_ms"]], [1.0], s=90, marker="*", color=col, zorder=5,
                   label=f"Kesin arama ({r['mode']})")
    ax.set_xscale("log")
    ax.set_xlabel("Sorgu başına süre (ms, log)")
    ax.set_ylabel("recall@5")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"ANN vs kesin arama, N={n:,}")
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    plotstyle.save(fig, save_path)


if __name__ == "__main__":
    main()
