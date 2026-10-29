"""
Phase 4: Final Arama Motoru

Hat (pipeline):  metin -> MiniLM embedding -> [TruncatedSVD (ops.)] -> L2 normalize
                 -> BLAS ile toplu matris çarpımı (Q @ C.T) -> argpartition top-k

Tasarım kararları (hepsi önceki aşamaların verisine dayanıyor):
  * Dense MiniLM, sparse TF-IDF değil (Phase 2A): 384 boyut x float32 = doküman
    başına 1.5 KB; anlamsal benzerlik (eş anlamlılar, parafraz) yakalar.
  * Normlar indekslemede BİR KEZ hesaplanır (Phase 1: profilde işin 2/3'ü norm idi).
  * Benzerlik tek BLAS sgemm çağrısı (Phase 2C: büyük işte en hızlı yol).
  * Top-k Numba prange ile paralel, O(N) (Phase 1: tam sıralama gereksiz;
    Phase 3: tek thread'li argpartition Amdahl'daki seri kısımdı).
  * Thread sayısı sabit 8 (Phase 3 taraması: hibrit CPU'da 16 thread BLAS'ı yavaşlatıyor).
  * float32: yarı bellek trafiği; doğruluk farkı ~1e-7 << 1e-5 tolerans.
  * Boyut indirgeme OPSİYONEL (svd_k): recall@5 kaybı ile hız kazancı arasındaki
    denge Phase 2B'de ölçüldü; varsayılan kesin (exact) arama.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase3_parallelism"))
from common import EMBED_MODEL, SEED, TOP_K, configure_threads, normalize_rows  # noqa: E402
from search_kernels import search_hybrid  # noqa: E402


class SimilaritySearchEngine:
    """Korpusu bir kez indeksleyen, sonra milisaniyeler içinde sorgu cevaplayan motor."""

    def __init__(self, corpus_emb, svd_k=None, threads=None):
        """
        İndeksleme (bir kerelik maliyet, arama benchmark'ına dahil değil):
          1. (ops.) TruncatedSVD ile 384 -> svd_k boyut
          2. Satırları normalize et (cosine = nokta çarpım)
          3. float32 + C-contiguous sakla (BLAS'ın en hızlı okuduğu yerleşim)
        """
        self.threads = configure_threads(threads)
        t0 = time.perf_counter()
        X = np.asarray(corpus_emb, dtype=np.float32)
        self.svd = None
        if svd_k:
            from sklearn.decomposition import TruncatedSVD
            # TruncatedSVD (PCA değil): ortalamayı çıkarmaz -> orijine göre açıyı,
            # yani cosine geometrisini korur. Phase 2B'de en hızlı yöntem.
            self.svd = TruncatedSVD(n_components=svd_k, random_state=SEED).fit(X)
            X = self.svd.transform(X).astype(np.float32)
        self.C = np.ascontiguousarray(normalize_rows(X).astype(np.float32))
        self.index_time_s = time.perf_counter() - t0
        self._model = None

    @property
    def n_docs(self):
        return self.C.shape[0]

    @property
    def dim(self):
        return self.C.shape[1]

    def prepare_queries(self, Q_emb):
        """Sorgu embedding'lerini korpusla AYNI dönüşümden geçirir (SVD + normalize)."""
        Q = np.atleast_2d(np.asarray(Q_emb, dtype=np.float32))
        if self.svd is not None:
            Q = self.svd.transform(Q).astype(np.float32)
        return np.ascontiguousarray(normalize_rows(Q).astype(np.float32))

    def search_prepared(self, Qn, k=TOP_K):
        """
        Hazırlanmış (normalize) sorgularla arama -- benchmark edilen sıcak yol.
        Dönüş: (idx [q x k], skor [q x k]), azalan skor sırası.
        """
        # Hibrit arama (Phase 3'ün kazananı):
        #   * S = Qn @ C.T  -> BLAS sgemm, GIL serbest, 8 thread
        #   * top-k         -> Numba prange ile satırlar paralel, tek geçiş O(N).
        #     np.argpartition tek thread'dir ve BLAS hızlandıktan sonra toplam
        #     sürenin çoğunu alıyordu (Amdahl'daki seri kısım).
        #   * çok sorguda bloklama: skor tamponu L3'e sığar, yeni bellek ayrılmaz.
        return search_hybrid(Qn, self.C, k)

    def search(self, Q_emb, k=TOP_K):
        """Ham embedding'lerle arama (dönüşüm + arama)."""
        return self.search_prepared(self.prepare_queries(Q_emb), k)

    # ---------------- Canlı demo için metin arayüzü ----------------
    def encode(self, texts):
        """Metni MiniLM ile embedding'e çevirir (model ilk çağrıda bir kez yüklenir)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBED_MODEL, device="cpu")
        return self._model.encode(list(texts), convert_to_numpy=True,
                                  normalize_embeddings=False).astype(np.float32)

    def search_text(self, texts, k=TOP_K):
        """Metin sorgu(lar)ı -> (idx, skor, embedding süresi, arama süresi)."""
        t0 = time.perf_counter()
        Q = self.encode(texts)
        t1 = time.perf_counter()
        idx, sc = self.search(Q, k)
        t2 = time.perf_counter()
        return idx, sc, t1 - t0, t2 - t1
