"""
dataset/embed.py -- Korpus ve sorgular için iki temsil üretir ve data/ altına önbelleğe alır.

  1. Dense: sentence-transformers all-MiniLM-L6-v2 (384 boyut, float32).
     Neden MiniLM? CPU'da hızlı (12K özet birkaç dakikada) ve 384 boyut proje
     şartındaki 128-768 aralığında. FAQ gereği embedding üretim süresi
     benchmark'a DAHİL DEĞİL; sadece bilgi amaçlı kaydediyoruz.
  2. Sparse: TF-IDF (scipy CSR). Kelime dağarcığı on binlerce boyut ama her
     özet sadece ~100 farklı kelime içerir -> %99+ sıfır. Phase 2A'da bu
     seyrekliğin bellek/hız etkisini ölçüyoruz.

Kullanım:  py -3.14 dataset/embed.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA_DIR, EMBED_MODEL, load_corpus, load_queries, save_json  # noqa: E402


def build_dense(corpus_texts, query_texts):
    """MiniLM ile dense embedding üretir (normalize ETMEDEN; normalizasyon arama sisteminin işi)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    t0 = time.perf_counter()
    C = model.encode(corpus_texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=False)
    Q = model.encode(query_texts, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=False)
    elapsed = time.perf_counter() - t0
    return C.astype(np.float32), Q.astype(np.float32), elapsed, model.max_seq_length


def build_sparse(corpus_texts, query_texts):
    """
    TF-IDF matrisini üretir.
      stop_words='english': "the, of, and" gibi her yerde geçen kelimeler benzerliğe katkı vermez.
      min_df=2: tek bir dokümanda geçen kelimeler (yazım hatası, nadir isim) sözlüğü şişirir.
      sublinear_tf: tf yerine 1+log(tf); bir kelimenin 10 kez geçmesi 10 kat önemli değildir.
      dtype=float32: dense taraf ile aynı hassasiyet -> adil bellek karşılaştırması.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5,
                          sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(corpus_texts).tocsr()
    Xq = vec.transform(query_texts).tocsr()
    return X, Xq, len(vec.vocabulary_)


def main():
    corpus = load_corpus()
    queries = load_queries()
    c_texts = [d["abstract"] for d in corpus]
    q_texts = [q["query"] for q in queries]
    print(f"Korpus: {len(c_texts)}, sorgu: {len(q_texts)}")

    C, Q, elapsed, max_len = build_dense(c_texts, q_texts)
    np.save(DATA_DIR / "corpus_embeddings.npy", C)
    np.save(DATA_DIR / "query_embeddings.npy", Q)
    print(f"Dense: {C.shape}, {elapsed:.1f} s")

    X, Xq, vocab = build_sparse(c_texts, q_texts)
    sparse.save_npz(DATA_DIR / "tfidf_corpus.npz", X)
    sparse.save_npz(DATA_DIR / "tfidf_queries.npz", Xq)
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"TF-IDF: {X.shape}, nnz={X.nnz}, yoğunluk={density:.5f}")

    save_json({
        "model": EMBED_MODEL, "dim": int(C.shape[1]), "max_seq_length": int(max_len),
        "n_corpus": int(C.shape[0]), "n_queries": int(Q.shape[0]),
        "embedding_time_s_not_benchmarked": elapsed,
        "tfidf_vocab": vocab, "tfidf_nnz": int(X.nnz), "tfidf_density": density,
        "tfidf_avg_nnz_per_doc": X.nnz / X.shape[0],
    }, "embedding_info.json")


if __name__ == "__main__":
    main()
