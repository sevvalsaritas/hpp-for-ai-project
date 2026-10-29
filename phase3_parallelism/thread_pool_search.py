"""
Phase 3: ThreadPoolExecutor Stratejisi (starter dosyası, tamamlandı)

Ne zaman doğru seçim? I/O-bound işler veya GIL'i SERBEST BIRAKAN işler (NumPy/BLAS).

Deney 1 (GIL serbest): Sorguları parçalara bölüp her parçayı bir thread'de
    BLAS ile ara. BLAS C kodudur ve çalışırken GIL'i bırakır -> thread'ler
    gerçekten paralel çalışır.
    Önemli: BLAS'ın kendi iç thread'lerini 1'e sabitliyoruz (threadpoolctl).
    Yoksa 16 Python thread'i x 16 BLAS thread'i = 256 thread -> aşırı abonelik
    (oversubscription): çekirdekler bağlam değiştirmekle vakit kaybeder.

Deney 2 (GIL tutulu): Aynı fikir saf Python cosine ile. Saf Python bytecode
    çalışırken GIL'i tutar -> aynı anda sadece BİR thread ilerler. Hızlanma
    beklemiyoruz (hatta thread geçiş yüküyle yavaşlama).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_kernels import chunk_ranges, search_block  # noqa: E402


def search_single_numpy(query, corpus, k=5):
    """Tek sorguyu NumPy ile arar (hocanın iskeleti; 'sorgu başına thread' sürümü için)."""
    similarities = query @ corpus.T
    top_k_idx = np.argpartition(-similarities, k)[:k]
    top_k_scores = similarities[top_k_idx]
    sorted_order = np.argsort(-top_k_scores)
    return list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order]))


def search_threadpool(queries, corpus, k=5, max_workers=4, executor=None):
    """
    Sorguları max_workers kadar PARÇAYA böler, her parçayı bir thread'de arar.

    Neden sorgu başına değil parça başına görev? 100 sorgu = 100 görev olsaydı
    her görev ~0.1 ms iş için future oluşturma/kuyruk yükü öderdi. Parça
    başına görev bu sabit yükü max_workers'a indirir (Amdahl: seri kısım küçülür).
    Ayrıca parça halinde BLAS, tek vektör (GEMV, BLAS-2) yerine matris (GEMM,
    BLAS-3) çarpımı yapar -> önbellekte veri tekrar kullanımı çok daha yüksek.
    """
    parts = chunk_ranges(len(queries), max_workers)
    own = executor is None
    ex = executor or ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [ex.submit(search_block, queries[a:b], corpus, k) for a, b in parts]
        res = [f.result() for f in futures]
    finally:
        if own:
            ex.shutdown()
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def search_threadpool_blas1(queries, corpus, k=5, max_workers=4):
    """search_threadpool'u BLAS iç thread'leri 1'e sabitlenmiş olarak çalıştırır (adil ölçüm)."""
    with threadpool_limits(limits=1):
        return search_threadpool(queries, corpus, k, max_workers)


def _python_cosine_block(q_list, c_list, k):
    """Saf Python arama (GIL'i tutar) -- thread'lerin GIL'e takıldığını göstermek için."""
    out = []
    for q in q_list:
        scores = []
        for j, c in enumerate(c_list):
            s = 0.0
            for a, b in zip(q, c):
                s += a * b
            scores.append((s, j))
        scores.sort(reverse=True)
        out.append([(j, s) for s, j in scores[:k]])
    return out


def search_threadpool_python(q_list, c_list, k=5, max_workers=4):
    """Saf Python işini thread'lere dağıtır. GIL nedeniyle hızlanma beklenmez."""
    parts = chunk_ranges(len(q_list), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_python_cosine_block, q_list[a:b], c_list, k) for a, b in parts]
        out = []
        for f in futures:
            out.extend(f.result())
    return out


if __name__ == "__main__":
    print("Bu strateji phase3_parallelism/run_phase3.py içinden ölçülür.")
